using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using Neuraval.Core.Training.Precision;

namespace Neuraval.Core.Serialization.SafeTensors
{
    public static class SafeTensorsWriter
    {
        public static void Write(
            string filePath,
            IReadOnlyList<SafeTensorsEntry> entries,
            IReadOnlyDictionary<string, string>? metadata = null)
        {
            if (filePath == null)
                throw new ArgumentNullException(nameof(filePath));

            if (entries == null)
                throw new ArgumentNullException(nameof(entries));

            if (entries.Count == 0)
                throw new ArgumentException("Se necesita al menos un tensor", nameof(entries));

            var seenNames = new HashSet<string>();
            foreach (var entry in entries)
            {
                if (!seenNames.Add(entry.Name))
                    throw new ArgumentException($"Nombre de tensor duplicado: '{entry.Name}'");
            }

            var directory = Path.GetDirectoryName(Path.GetFullPath(filePath));
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            byte[] body = BuildBody(entries, out var offsets);
            byte[] headerBytes = BuildHeaderJson(entries, offsets, metadata);

            using var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
            using var writer = new BinaryWriter(stream);

            Span<byte> headerSizeBytes = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(headerSizeBytes, (ulong)headerBytes.Length);

            writer.Write(headerSizeBytes.ToArray());
            writer.Write(headerBytes);
            writer.Write(body);
        }

        private static byte[] BuildBody(IReadOnlyList<SafeTensorsEntry> entries, out (long Start, long End)[] offsets)
        {
            offsets = new (long, long)[entries.Count];

            long totalBytes = 0;
            for (int i = 0; i < entries.Count; i++)
            {
                long size = (long)entries[i].Data.Length * entries[i].DType.ByteSize();
                offsets[i] = (totalBytes, totalBytes + size);
                totalBytes += size;
            }

            var body = new byte[totalBytes];

            for (int i = 0; i < entries.Count; i++)
            {
                WriteTensorBytes(entries[i], body.AsSpan((int)offsets[i].Start, entries[i].Data.Length * entries[i].DType.ByteSize()));
            }

            return body;
        }

        private static void WriteTensorBytes(SafeTensorsEntry entry, Span<byte> destination)
        {
            switch (entry.DType)
            {
                case SafeTensorsDType.F32:
                    for (int i = 0; i < entry.Data.Length; i++)
                    {
                        BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(i * 4, 4), entry.Data[i]);
                    }
                    break;

                case SafeTensorsDType.F16:
                    for (int i = 0; i < entry.Data.Length; i++)
                    {
                        ushort bits = BitConverter.HalfToUInt16Bits((Half)entry.Data[i]);
                        BinaryPrimitives.WriteUInt16LittleEndian(destination.Slice(i * 2, 2), bits);
                    }
                    break;

                case SafeTensorsDType.BF16:
                    for (int i = 0; i < entry.Data.Length; i++)
                    {
                        ushort bits = BFloat16.FromSingle(entry.Data[i]).RawBits;
                        BinaryPrimitives.WriteUInt16LittleEndian(destination.Slice(i * 2, 2), bits);
                    }
                    break;

                default:
                    throw new ArgumentOutOfRangeException(nameof(entry));
            }
        }

        private static byte[] BuildHeaderJson(
            IReadOnlyList<SafeTensorsEntry> entries,
            (long Start, long End)[] offsets,
            IReadOnlyDictionary<string, string>? metadata)
        {
            using var stream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(stream))
            {
                writer.WriteStartObject();

                if (metadata != null && metadata.Count > 0)
                {
                    writer.WriteStartObject("__metadata__");
                    foreach (var kv in metadata)
                    {
                        writer.WriteString(kv.Key, kv.Value);
                    }
                    writer.WriteEndObject();
                }

                for (int i = 0; i < entries.Count; i++)
                {
                    writer.WriteStartObject(entries[i].Name);

                    writer.WriteString("dtype", entries[i].DType.ToTag());

                    writer.WriteStartArray("shape");
                    foreach (var dim in entries[i].Shape)
                    {
                        writer.WriteNumberValue(dim);
                    }
                    writer.WriteEndArray();

                    writer.WriteStartArray("data_offsets");
                    writer.WriteNumberValue(offsets[i].Start);
                    writer.WriteNumberValue(offsets[i].End);
                    writer.WriteEndArray();

                    writer.WriteEndObject();
                }

                writer.WriteEndObject();
            }

            return stream.ToArray();
        }
    }
}
