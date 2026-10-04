using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Neuraval.Core.Serialization.Gguf
{
    public static class GgufReader
    {
        private const uint MagicValue = 0x46554747;
        private const int DefaultAlignment = 32;

        public static GgufFile Read(string filePath)
        {
            if (filePath == null)
                throw new ArgumentNullException(nameof(filePath));

            var buffer = File.ReadAllBytes(filePath);
            return Read(buffer);
        }

        public static GgufFile Read(byte[] buffer)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));

            var cursor = new Cursor(buffer);

            uint magic = cursor.ReadUInt32();
            if (magic != MagicValue)
                throw new InvalidDataException("El archivo no tiene la firma GGUF esperada");

            uint version = cursor.ReadUInt32();
            if (version < 2)
                throw new NotSupportedException($"Version GGUF {version} no soportada, se requiere version 2 o superior");

            ulong tensorCount = cursor.ReadUInt64();
            ulong metadataCount = cursor.ReadUInt64();

            var metadata = new Dictionary<string, GgufMetadataValue>();
            for (ulong i = 0; i < metadataCount; i++)
            {
                string key = cursor.ReadString();
                var value = ReadValue(cursor);
                metadata[key] = value;
            }

            int alignment = DefaultAlignment;
            if (metadata.TryGetValue("general.alignment", out var alignmentValue) && alignmentValue.TryGetUInt64(out var alignmentRaw))
                alignment = (int)alignmentRaw;

            var tensorInfos = new List<TensorInfo>();
            for (ulong i = 0; i < tensorCount; i++)
            {
                string name = cursor.ReadString();
                uint dimensionCount = cursor.ReadUInt32();

                var dimensions = new ulong[dimensionCount];
                for (uint d = 0; d < dimensionCount; d++)
                    dimensions[d] = cursor.ReadUInt64();

                uint rawType = cursor.ReadUInt32();
                ulong offset = cursor.ReadUInt64();

                tensorInfos.Add(new TensorInfo(name, dimensions, rawType, offset));
            }

            int dataStart = AlignUp(cursor.Position, alignment);

            var tensors = new List<GgufTensorEntry>();
            foreach (var info in tensorInfos)
            {
                int absoluteOffset = dataStart + checked((int)info.Offset);
                var entry = DecodeTensor(buffer, absoluteOffset, info);
                tensors.Add(entry);
            }

            return new GgufFile(version, metadata, tensors);
        }

        private static GgufMetadataValue ReadValue(Cursor cursor)
        {
            var type = (GgufValueType)cursor.ReadUInt32();
            return new GgufMetadataValue(type, ReadRaw(cursor, type));
        }

        private static object ReadRaw(Cursor cursor, GgufValueType type)
        {
            switch (type)
            {
                case GgufValueType.UInt8: return cursor.ReadByte();
                case GgufValueType.Int8: return cursor.ReadSByte();
                case GgufValueType.UInt16: return cursor.ReadUInt16();
                case GgufValueType.Int16: return cursor.ReadInt16();
                case GgufValueType.UInt32: return cursor.ReadUInt32();
                case GgufValueType.Int32: return cursor.ReadInt32();
                case GgufValueType.Float32: return cursor.ReadFloat32();
                case GgufValueType.Bool: return cursor.ReadByte() != 0;
                case GgufValueType.String: return cursor.ReadString();
                case GgufValueType.UInt64: return cursor.ReadUInt64();
                case GgufValueType.Int64: return cursor.ReadInt64();
                case GgufValueType.Float64: return cursor.ReadFloat64();
                case GgufValueType.Array: return ReadArray(cursor);
                default: throw new NotSupportedException($"Tipo de metadata GGUF '{type}' no soportado");
            }
        }

        private static IReadOnlyList<GgufMetadataValue> ReadArray(Cursor cursor)
        {
            var elementType = (GgufValueType)cursor.ReadUInt32();
            ulong length = cursor.ReadUInt64();

            var elements = new List<GgufMetadataValue>((int)Math.Min(length, int.MaxValue));
            for (ulong i = 0; i < length; i++)
                elements.Add(new GgufMetadataValue(elementType, ReadRaw(cursor, elementType)));

            return elements;
        }

        private static GgufTensorEntry DecodeTensor(byte[] buffer, int absoluteOffset, TensorInfo info)
        {
            var shape = ToRowMajorShape(info.Dimensions);
            long elementCount = 1;
            foreach (var dim in shape)
                elementCount *= dim;

            var type = (GgmlType)info.RawType;
            float[] data = Dequantize(buffer, absoluteOffset, type, elementCount, info.Name);

            return new GgufTensorEntry(info.Name, type, shape, data);
        }

        private static int[] ToRowMajorShape(ulong[] dimensions)
        {
            var shape = new int[dimensions.Length];
            for (int i = 0; i < dimensions.Length; i++)
                shape[i] = checked((int)dimensions[dimensions.Length - 1 - i]);

            return shape;
        }

        private static float[] Dequantize(byte[] buffer, int offset, GgmlType type, long elementCount, string tensorName)
        {
            switch (type)
            {
                case GgmlType.F32:
                    return DequantizeF32(buffer, offset, elementCount);
                case GgmlType.F16:
                    return DequantizeF16(buffer, offset, elementCount);
                case GgmlType.Q8_0:
                    return DequantizeQ8_0(buffer, offset, elementCount);
                case GgmlType.Q4_0:
                    return DequantizeQ4_0(buffer, offset, elementCount);
                case GgmlType.Q3_K:
                    return DequantizeQ3_K(buffer, offset, elementCount);
                case GgmlType.Q2_K:
                    return DequantizeQ2_K(buffer, offset, elementCount);
                case GgmlType.Q4_K:
                    return DequantizeQ4_K(buffer, offset, elementCount);
                case GgmlType.Q5_K:
                    return DequantizeQ5_K(buffer, offset, elementCount);
                case GgmlType.Q6_K:
                    return DequantizeQ6_K(buffer, offset, elementCount);
                case GgmlType.Q8_K:
                    return DequantizeQ8_K(buffer, offset, elementCount);
                case GgmlType.IQ4_NL:
                    return DequantizeIq4Nl(buffer, offset, elementCount);
                default:
                    throw new NotSupportedException(
                        $"El tensor '{tensorName}' usa el tipo GGML '{type}' (id {(uint)type}), que no esta soportado por el loader de Neuraval");
            }
        }

        private static float[] DequantizeF32(byte[] buffer, int offset, long elementCount)
        {
            var result = new float[elementCount];
            for (long i = 0; i < elementCount; i++)
                result[i] = BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(offset + (int)(i * 4), 4));

            return result;
        }

        private static float[] DequantizeF16(byte[] buffer, int offset, long elementCount)
        {
            var result = new float[elementCount];
            for (long i = 0; i < elementCount; i++)
            {
                ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(offset + (int)(i * 2), 2));
                result[i] = (float)BitConverter.UInt16BitsToHalf(bits);
            }

            return result;
        }

        private static float[] DequantizeQ8_0(byte[] buffer, int offset, long elementCount)
        {
            const int blockElements = 32;
            const int blockBytes = 2 + blockElements;

            if (elementCount % blockElements != 0)
                throw new InvalidDataException("El numero de elementos Q8_0 debe ser multiplo de 32");

            long blockCount = elementCount / blockElements;
            var result = new float[elementCount];

            for (long block = 0; block < blockCount; block++)
            {
                int blockOffset = offset + (int)(block * blockBytes);
                ushort scaleBits = BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(blockOffset, 2));
                float scale = (float)BitConverter.UInt16BitsToHalf(scaleBits);

                for (int j = 0; j < blockElements; j++)
                {
                    sbyte quantized = unchecked((sbyte)buffer[blockOffset + 2 + j]);
                    result[block * blockElements + j] = quantized * scale;
                }
            }

            return result;
        }

        private static float[] DequantizeQ4_0(byte[] buffer, int offset, long elementCount)
        {
            const int blockElements = 32;
            const int halfBlockElements = blockElements / 2;
            const int blockBytes = 2 + halfBlockElements;

            if (elementCount % blockElements != 0)
                throw new InvalidDataException("El numero de elementos Q4_0 debe ser multiplo de 32");

            long blockCount = elementCount / blockElements;
            var result = new float[elementCount];

            for (long block = 0; block < blockCount; block++)
            {
                int blockOffset = offset + (int)(block * blockBytes);
                ushort scaleBits = BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(blockOffset, 2));
                float scale = (float)BitConverter.UInt16BitsToHalf(scaleBits);

                for (int j = 0; j < halfBlockElements; j++)
                {
                    byte packed = buffer[blockOffset + 2 + j];
                    int low = (packed & 0x0F) - 8;
                    int high = (packed >> 4) - 8;

                    result[block * blockElements + j] = low * scale;
                    result[block * blockElements + j + halfBlockElements] = high * scale;
                }
            }

            return result;
        }

        private static float[] DequantizeQ3_K(byte[] buffer, int offset, long elementCount)
        {
            const int blockElements = 256;
            const int hmaskSize = 32;
            const int qsSize = 64;
            const int scalesSize = 12;
            const int blockBytes = hmaskSize + qsSize + scalesSize + 2;

            const uint kmask1 = 0x03030303;
            const uint kmask2 = 0x0f0f0f0f;

            if (elementCount % blockElements != 0)
                throw new InvalidDataException("El numero de elementos Q3_K debe ser multiplo de 256");

            long blockCount = elementCount / blockElements;
            var result = new float[elementCount];
            Span<byte> scaleBytes = stackalloc byte[16];

            for (long block = 0; block < blockCount; block++)
            {
                int blockOffset = offset + (int)(block * blockBytes);
                int hmBase = blockOffset;
                int qBase0 = hmBase + hmaskSize;
                int scalesOffset = qBase0 + qsSize;
                int dOffset = scalesOffset + scalesSize;

                float dAll = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(dOffset, 2)));

                DecodeQ3KScales(buffer, scalesOffset, kmask1, kmask2, scaleBytes);

                long outBase = block * blockElements;
                int outCursor = 0;
                int is_ = 0;
                byte m = 1;
                int qBase = qBase0;

                for (int n = 0; n < blockElements; n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; j++)
                    {
                        float dl1 = dAll * (unchecked((sbyte)scaleBytes[is_]) - 32);
                        is_++;
                        for (int l = 0; l < 16; l++)
                        {
                            int q3 = (buffer[qBase + l] >> shift) & 3;
                            int highBitOff = (buffer[hmBase + l] & m) != 0 ? 0 : 4;
                            result[outBase + outCursor] = dl1 * (q3 - highBitOff);
                            outCursor++;
                        }

                        float dl2 = dAll * (unchecked((sbyte)scaleBytes[is_]) - 32);
                        is_++;
                        for (int l = 0; l < 16; l++)
                        {
                            int q3 = (buffer[qBase + l + 16] >> shift) & 3;
                            int highBitOff = (buffer[hmBase + l + 16] & m) != 0 ? 0 : 4;
                            result[outBase + outCursor] = dl2 * (q3 - highBitOff);
                            outCursor++;
                        }

                        shift += 2;
                        m = (byte)(m << 1);
                    }

                    qBase += 32;
                }
            }

            return result;
        }

        private static float[] DequantizeQ2_K(byte[] buffer, int offset, long elementCount)
        {
            const int blockElements = 256;
            const int scalesSize = 16;
            const int qsSize = 64;
            const int blockBytes = scalesSize + qsSize + 4;

            if (elementCount % blockElements != 0)
                throw new InvalidDataException("El numero de elementos Q2_K debe ser multiplo de 256");

            long blockCount = elementCount / blockElements;
            var result = new float[elementCount];

            for (long block = 0; block < blockCount; block++)
            {
                int blockOffset = offset + (int)(block * blockBytes);
                int scalesBase = blockOffset;
                int qBase0 = scalesBase + scalesSize;
                int dOffset = qBase0 + qsSize;
                int dminOffset = dOffset + 2;

                float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(dOffset, 2)));
                float dmin = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(dminOffset, 2)));

                long outBase = block * blockElements;
                int outCursor = 0;
                int is_ = 0;
                int qBase = qBase0;

                for (int n = 0; n < blockElements; n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; j++)
                    {
                        byte sc1 = buffer[scalesBase + is_];
                        is_++;
                        float dl1 = d * (sc1 & 0x0F);
                        float ml1 = dmin * (sc1 >> 4);
                        for (int l = 0; l < 16; l++)
                        {
                            int q2 = (buffer[qBase + l] >> shift) & 3;
                            result[outBase + outCursor] = dl1 * q2 - ml1;
                            outCursor++;
                        }

                        byte sc2 = buffer[scalesBase + is_];
                        is_++;
                        float dl2 = d * (sc2 & 0x0F);
                        float ml2 = dmin * (sc2 >> 4);
                        for (int l = 0; l < 16; l++)
                        {
                            int q2 = (buffer[qBase + l + 16] >> shift) & 3;
                            result[outBase + outCursor] = dl2 * q2 - ml2;
                            outCursor++;
                        }

                        shift += 2;
                    }

                    qBase += 32;
                }
            }

            return result;
        }

        private static void DecodeQ3KScales(byte[] buffer, int scalesOffset, uint kmask1, uint kmask2, Span<byte> scaleBytes)
        {
            uint aux0 = BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(scalesOffset, 4));
            uint aux1 = BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(scalesOffset + 4, 4));
            uint tmp = BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(scalesOffset + 8, 4));

            uint newAux0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
            uint newAux1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
            uint newAux2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            uint newAux3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);

            BinaryPrimitives.WriteUInt32LittleEndian(scaleBytes.Slice(0, 4), newAux0);
            BinaryPrimitives.WriteUInt32LittleEndian(scaleBytes.Slice(4, 4), newAux1);
            BinaryPrimitives.WriteUInt32LittleEndian(scaleBytes.Slice(8, 4), newAux2);
            BinaryPrimitives.WriteUInt32LittleEndian(scaleBytes.Slice(12, 4), newAux3);
        }

        private static readonly sbyte[] Iq4NlValues =
            { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

        private static void GetScaleMinK4(byte[] buffer, int scalesOffset, int j, out byte scale, out byte min)
        {
            if (j < 4)
            {
                scale = (byte)(buffer[scalesOffset + j] & 0x3F);
                min = (byte)(buffer[scalesOffset + j + 4] & 0x3F);
            }
            else
            {
                scale = (byte)((buffer[scalesOffset + j + 4] & 0x0F) | ((buffer[scalesOffset + j - 4] >> 6) << 4));
                min = (byte)((buffer[scalesOffset + j + 4] >> 4) | ((buffer[scalesOffset + j] >> 6) << 4));
            }
        }

        private static float[] DequantizeQ4_K(byte[] buffer, int offset, long elementCount)
        {
            const int blockElements = 256;
            const int scalesSize = 12;
            const int qsSize = 128;
            const int blockBytes = 4 + scalesSize + qsSize;

            if (elementCount % blockElements != 0)
                throw new InvalidDataException("El numero de elementos Q4_K debe ser multiplo de 256");

            long blockCount = elementCount / blockElements;
            var result = new float[elementCount];

            for (long block = 0; block < blockCount; block++)
            {
                int blockOffset = offset + (int)(block * blockBytes);
                float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(blockOffset, 2)));
                float dmin = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(blockOffset + 2, 2)));

                int scalesOffset = blockOffset + 4;
                int qsOffset = scalesOffset + scalesSize;
                long outBase = block * blockElements;

                int subBlock = 0;
                int qsCursor = 0;
                for (int j = 0; j < blockElements; j += 64)
                {
                    GetScaleMinK4(buffer, scalesOffset, subBlock + 0, out byte sc1, out byte m1);
                    GetScaleMinK4(buffer, scalesOffset, subBlock + 1, out byte sc2, out byte m2);
                    float d1 = d * sc1;
                    float mm1 = dmin * m1;
                    float d2 = d * sc2;
                    float mm2 = dmin * m2;

                    for (int l = 0; l < 32; l++)
                    {
                        byte packed = buffer[qsOffset + qsCursor + l];
                        result[outBase + j + l] = d1 * (packed & 0x0F) - mm1;
                        result[outBase + j + 32 + l] = d2 * (packed >> 4) - mm2;
                    }

                    qsCursor += 32;
                    subBlock += 2;
                }
            }

            return result;
        }

        private static float[] DequantizeQ5_K(byte[] buffer, int offset, long elementCount)
        {
            const int blockElements = 256;
            const int scalesSize = 12;
            const int qhSize = 32;
            const int qsSize = 128;
            const int blockBytes = 4 + scalesSize + qhSize + qsSize;

            if (elementCount % blockElements != 0)
                throw new InvalidDataException("El numero de elementos Q5_K debe ser multiplo de 256");

            long blockCount = elementCount / blockElements;
            var result = new float[elementCount];

            for (long block = 0; block < blockCount; block++)
            {
                int blockOffset = offset + (int)(block * blockBytes);
                float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(blockOffset, 2)));
                float dmin = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(blockOffset + 2, 2)));

                int scalesOffset = blockOffset + 4;
                int qhOffset = scalesOffset + scalesSize;
                int qsOffset = qhOffset + qhSize;
                long outBase = block * blockElements;

                int subBlock = 0;
                int qsCursor = 0;
                byte u1 = 1;
                byte u2 = 2;

                for (int j = 0; j < blockElements; j += 64)
                {
                    GetScaleMinK4(buffer, scalesOffset, subBlock + 0, out byte sc1, out byte m1);
                    GetScaleMinK4(buffer, scalesOffset, subBlock + 1, out byte sc2, out byte m2);
                    float d1 = d * sc1;
                    float mm1 = dmin * m1;
                    float d2 = d * sc2;
                    float mm2 = dmin * m2;

                    for (int l = 0; l < 32; l++)
                    {
                        byte ql = buffer[qsOffset + qsCursor + l];
                        byte qh = buffer[qhOffset + l];
                        int highBit1 = (qh & u1) != 0 ? 16 : 0;
                        int highBit2 = (qh & u2) != 0 ? 16 : 0;

                        result[outBase + j + l] = d1 * ((ql & 0x0F) + highBit1) - mm1;
                        result[outBase + j + 32 + l] = d2 * ((ql >> 4) + highBit2) - mm2;
                    }

                    qsCursor += 32;
                    subBlock += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }

            return result;
        }

        private static float[] DequantizeQ6_K(byte[] buffer, int offset, long elementCount)
        {
            const int blockElements = 256;
            const int qlSize = 128;
            const int qhSize = 64;
            const int scalesSize = 16;
            const int blockBytes = qlSize + qhSize + scalesSize + 2;

            if (elementCount % blockElements != 0)
                throw new InvalidDataException("El numero de elementos Q6_K debe ser multiplo de 256");

            long blockCount = elementCount / blockElements;
            var result = new float[elementCount];

            for (long block = 0; block < blockCount; block++)
            {
                int blockOffset = offset + (int)(block * blockBytes);
                int qlBase = blockOffset;
                int qhBase = qlBase + qlSize;
                int scalesBase = qhBase + qhSize;
                int dOffset = scalesBase + scalesSize;

                float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(dOffset, 2)));
                long outBase = block * blockElements;

                for (int n = 0; n < blockElements; n += 128)
                {
                    int chunk = n / 128;
                    int qlOffset = qlBase + chunk * 64;
                    int qhOffset = qhBase + chunk * 32;
                    int scOffset = scalesBase + chunk * 8;

                    for (int l = 0; l < 32; l++)
                    {
                        int isIdx = l / 16;
                        int ql0 = buffer[qlOffset + l];
                        int ql32 = buffer[qlOffset + l + 32];
                        int qh = buffer[qhOffset + l];

                        int q1 = ((ql0 & 0x0F) | (((qh >> 0) & 3) << 4)) - 32;
                        int q2 = ((ql32 & 0x0F) | (((qh >> 2) & 3) << 4)) - 32;
                        int q3 = ((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32;
                        int q4 = ((ql32 >> 4) | (((qh >> 6) & 3) << 4)) - 32;

                        sbyte sc0 = unchecked((sbyte)buffer[scOffset + isIdx + 0]);
                        sbyte sc2 = unchecked((sbyte)buffer[scOffset + isIdx + 2]);
                        sbyte sc4 = unchecked((sbyte)buffer[scOffset + isIdx + 4]);
                        sbyte sc6 = unchecked((sbyte)buffer[scOffset + isIdx + 6]);

                        result[outBase + n + l] = d * sc0 * q1;
                        result[outBase + n + l + 32] = d * sc2 * q2;
                        result[outBase + n + l + 64] = d * sc4 * q3;
                        result[outBase + n + l + 96] = d * sc6 * q4;
                    }
                }
            }

            return result;
        }

        private static float[] DequantizeQ8_K(byte[] buffer, int offset, long elementCount)
        {
            const int blockElements = 256;
            const int qsSize = 256;
            const int bsumsSize = 32;
            const int blockBytes = 4 + qsSize + bsumsSize;

            if (elementCount % blockElements != 0)
                throw new InvalidDataException("El numero de elementos Q8_K debe ser multiplo de 256");

            long blockCount = elementCount / blockElements;
            var result = new float[elementCount];

            for (long block = 0; block < blockCount; block++)
            {
                int blockOffset = offset + (int)(block * blockBytes);
                float d = BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(blockOffset, 4));
                int qsOffset = blockOffset + 4;
                long outBase = block * blockElements;

                for (int j = 0; j < blockElements; j++)
                {
                    sbyte q = unchecked((sbyte)buffer[qsOffset + j]);
                    result[outBase + j] = d * q;
                }
            }

            return result;
        }

        private static float[] DequantizeIq4Nl(byte[] buffer, int offset, long elementCount)
        {
            const int blockElements = 32;
            const int halfBlockElements = blockElements / 2;
            const int blockBytes = 2 + halfBlockElements;

            if (elementCount % blockElements != 0)
                throw new InvalidDataException("El numero de elementos IQ4_NL debe ser multiplo de 32");

            long blockCount = elementCount / blockElements;
            var result = new float[elementCount];

            for (long block = 0; block < blockCount; block++)
            {
                int blockOffset = offset + (int)(block * blockBytes);
                float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(buffer.AsSpan(blockOffset, 2)));
                int qsOffset = blockOffset + 2;
                long outBase = block * blockElements;

                for (int j = 0; j < halfBlockElements; j++)
                {
                    byte packed = buffer[qsOffset + j];
                    result[outBase + j] = d * Iq4NlValues[packed & 0x0F];
                    result[outBase + j + halfBlockElements] = d * Iq4NlValues[packed >> 4];
                }
            }

            return result;
        }

        private static int AlignUp(int value, int alignment)
        {
            int remainder = value % alignment;
            return remainder == 0 ? value : value + (alignment - remainder);
        }

        private sealed class TensorInfo
        {
            public string Name { get; }
            public ulong[] Dimensions { get; }
            public uint RawType { get; }
            public ulong Offset { get; }

            public TensorInfo(string name, ulong[] dimensions, uint rawType, ulong offset)
            {
                Name = name;
                Dimensions = dimensions;
                RawType = rawType;
                Offset = offset;
            }
        }

        private sealed class Cursor
        {
            private readonly byte[] _buffer;
            private int _position;

            public Cursor(byte[] buffer)
            {
                _buffer = buffer;
                _position = 0;
            }

            public int Position => _position;

            public byte ReadByte()
            {
                var value = _buffer[_position];
                _position += 1;
                return value;
            }

            public sbyte ReadSByte()
            {
                return unchecked((sbyte)ReadByte());
            }

            public ushort ReadUInt16()
            {
                var value = BinaryPrimitives.ReadUInt16LittleEndian(_buffer.AsSpan(_position, 2));
                _position += 2;
                return value;
            }

            public short ReadInt16()
            {
                var value = BinaryPrimitives.ReadInt16LittleEndian(_buffer.AsSpan(_position, 2));
                _position += 2;
                return value;
            }

            public uint ReadUInt32()
            {
                var value = BinaryPrimitives.ReadUInt32LittleEndian(_buffer.AsSpan(_position, 4));
                _position += 4;
                return value;
            }

            public int ReadInt32()
            {
                var value = BinaryPrimitives.ReadInt32LittleEndian(_buffer.AsSpan(_position, 4));
                _position += 4;
                return value;
            }

            public ulong ReadUInt64()
            {
                var value = BinaryPrimitives.ReadUInt64LittleEndian(_buffer.AsSpan(_position, 8));
                _position += 8;
                return value;
            }

            public long ReadInt64()
            {
                var value = BinaryPrimitives.ReadInt64LittleEndian(_buffer.AsSpan(_position, 8));
                _position += 8;
                return value;
            }

            public float ReadFloat32()
            {
                var value = BinaryPrimitives.ReadSingleLittleEndian(_buffer.AsSpan(_position, 4));
                _position += 4;
                return value;
            }

            public double ReadFloat64()
            {
                var value = BinaryPrimitives.ReadDoubleLittleEndian(_buffer.AsSpan(_position, 8));
                _position += 8;
                return value;
            }

            public string ReadString()
            {
                ulong length = ReadUInt64();
                var text = Encoding.UTF8.GetString(_buffer, _position, checked((int)length));
                _position += checked((int)length);
                return text;
            }
        }
    }
}
