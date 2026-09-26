using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace Neuraval.Core.Serialization.SafeTensors
{
    public static class SafeTensorsReader
    {
        public static SafeTensorsFile Read(string filePath)
        {
            if (filePath == null)
                throw new ArgumentNullException(nameof(filePath));

            using var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            return Read(stream);
        }

        public static SafeTensorsFile Read(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            Span<byte> headerSizeBytes = stackalloc byte[8];
            ReadExact(stream, headerSizeBytes);
            ulong headerSize = BinaryPrimitives.ReadUInt64LittleEndian(headerSizeBytes);

            var headerBytes = new byte[headerSize];
            ReadExact(stream, headerBytes);

            using var body = new MemoryStream();
            stream.CopyTo(body);
            byte[] bodyBytes = body.ToArray();

            using var document = JsonDocument.Parse(headerBytes);

            var metadata = new Dictionary<string, string>();
            var tensors = new List<SafeTensorsEntry>();

            foreach (var property in document.RootElement.EnumerateObject())
            {
                if (property.Name == "__metadata__")
                {
                    foreach (var metaProperty in property.Value.EnumerateObject())
                    {
                        metadata[metaProperty.Name] = metaProperty.Value.GetString() ?? string.Empty;
                    }
                    continue;
                }

                var dtype = SafeTensorsDTypeExtensions.FromTag(property.Value.GetProperty("dtype").GetString()!);

                var shapeElement = property.Value.GetProperty("shape");
                var shape = new int[shapeElement.GetArrayLength()];
                int shapeIndex = 0;
                foreach (var dim in shapeElement.EnumerateArray())
                {
                    shape[shapeIndex++] = dim.GetInt32();
                }

                var offsetsElement = property.Value.GetProperty("data_offsets");
                var offsetEnumerator = offsetsElement.EnumerateArray().GetEnumerator();
                offsetEnumerator.MoveNext();
                long start = offsetEnumerator.Current.GetInt64();
                offsetEnumerator.MoveNext();
                long end = offsetEnumerator.Current.GetInt64();

                var slice = bodyBytes.AsSpan((int)start, (int)(end - start));
                var data = DecodeTensorBytes(dtype, slice);

                tensors.Add(new SafeTensorsEntry(property.Name, dtype, shape, data));
            }

            return new SafeTensorsFile(tensors, metadata);
        }

        private static float[] DecodeTensorBytes(SafeTensorsDType dtype, ReadOnlySpan<byte> source)
        {
            int elementSize = dtype.ByteSize();
            int count = source.Length / elementSize;
            var data = new float[count];

            switch (dtype)
            {
                case SafeTensorsDType.F32:
                    for (int i = 0; i < count; i++)
                    {
                        data[i] = BinaryPrimitives.ReadSingleLittleEndian(source.Slice(i * 4, 4));
                    }
                    break;

                case SafeTensorsDType.F16:
                    for (int i = 0; i < count; i++)
                    {
                        ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(source.Slice(i * 2, 2));
                        data[i] = (float)BitConverter.UInt16BitsToHalf(bits);
                    }
                    break;

                case SafeTensorsDType.BF16:
                    for (int i = 0; i < count; i++)
                    {
                        ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(source.Slice(i * 2, 2));
                        data[i] = BitConverter.UInt32BitsToSingle((uint)bits << 16);
                    }
                    break;

                default:
                    throw new ArgumentOutOfRangeException(nameof(dtype));
            }

            return data;
        }

        private static void ReadExact(Stream stream, Span<byte> destination)
        {
            int totalRead = 0;
            while (totalRead < destination.Length)
            {
                int read = stream.Read(destination.Slice(totalRead));
                if (read == 0)
                    throw new EndOfStreamException("Archivo safetensors truncado");

                totalRead += read;
            }
        }
    }
}
