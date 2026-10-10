using System;
using System.Collections.Generic;
using System.IO;
using Neuraval.Core.Serialization.Gguf;
using Xunit;

namespace Neuraval.Tests
{
    public class GgufReaderTests
    {
        private sealed class TestTensor
        {
            public string Name { get; }
            public GgmlType Type { get; }
            public int[] Shape { get; }
            public byte[] RawBytes { get; }

            public TestTensor(string name, GgmlType type, int[] shape, byte[] rawBytes)
            {
                Name = name;
                Type = type;
                Shape = shape;
                RawBytes = rawBytes;
            }
        }

        private sealed class GgufBuilder
        {
            private readonly List<(string Key, GgufValueType Type, byte[] Bytes)> _metadata = new();
            private readonly List<TestTensor> _tensors = new();
            private uint _version = 3;
            private int _magicOverride = -1;

            public GgufBuilder WithVersion(uint version)
            {
                _version = version;
                return this;
            }

            public GgufBuilder WithBadMagic()
            {
                _magicOverride = 0;
                return this;
            }

            public GgufBuilder AddString(string key, string value)
            {
                _metadata.Add((key, GgufValueType.String, EncodeString(value)));
                return this;
            }

            public GgufBuilder AddUInt32(string key, uint value)
            {
                _metadata.Add((key, GgufValueType.UInt32, BitConverter.GetBytes(value)));
                return this;
            }

            public GgufBuilder AddInt32(string key, int value)
            {
                _metadata.Add((key, GgufValueType.Int32, BitConverter.GetBytes(value)));
                return this;
            }

            public GgufBuilder AddUInt64(string key, ulong value)
            {
                _metadata.Add((key, GgufValueType.UInt64, BitConverter.GetBytes(value)));
                return this;
            }

            public GgufBuilder AddInt64(string key, long value)
            {
                _metadata.Add((key, GgufValueType.Int64, BitConverter.GetBytes(value)));
                return this;
            }

            public GgufBuilder AddFloat32(string key, float value)
            {
                _metadata.Add((key, GgufValueType.Float32, BitConverter.GetBytes(value)));
                return this;
            }

            public GgufBuilder AddFloat64(string key, double value)
            {
                _metadata.Add((key, GgufValueType.Float64, BitConverter.GetBytes(value)));
                return this;
            }

            public GgufBuilder AddBool(string key, bool value)
            {
                _metadata.Add((key, GgufValueType.Bool, new[] { (byte)(value ? 1 : 0) }));
                return this;
            }

            public GgufBuilder AddUInt8(string key, byte value)
            {
                _metadata.Add((key, GgufValueType.UInt8, new[] { value }));
                return this;
            }

            public GgufBuilder AddInt8(string key, sbyte value)
            {
                _metadata.Add((key, GgufValueType.Int8, new[] { unchecked((byte)value) }));
                return this;
            }

            public GgufBuilder AddUInt16(string key, ushort value)
            {
                _metadata.Add((key, GgufValueType.UInt16, BitConverter.GetBytes(value)));
                return this;
            }

            public GgufBuilder AddInt16(string key, short value)
            {
                _metadata.Add((key, GgufValueType.Int16, BitConverter.GetBytes(value)));
                return this;
            }

            public GgufBuilder AddStringArray(string key, params string[] values)
            {
                using var stream = new MemoryStream();
                WriteUInt32(stream, (uint)GgufValueType.String);
                WriteUInt64(stream, (ulong)values.Length);
                foreach (var value in values)
                    stream.Write(EncodeString(value));

                _metadata.Add((key, GgufValueType.Array, stream.ToArray()));
                return this;
            }

            public GgufBuilder AddTensor(string name, GgmlType type, int[] shape, byte[] rawBytes)
            {
                _tensors.Add(new TestTensor(name, type, shape, rawBytes));
                return this;
            }

            public byte[] Build(int alignment = 32)
            {
                using var stream = new MemoryStream();

                if (_magicOverride == 0)
                    stream.Write(new byte[] { 0, 0, 0, 0 });
                else
                    stream.Write(new byte[] { (byte)'G', (byte)'G', (byte)'U', (byte)'F' });

                WriteUInt32(stream, _version);
                WriteUInt64(stream, (ulong)_tensors.Count);
                WriteUInt64(stream, (ulong)_metadata.Count);

                foreach (var entry in _metadata)
                {
                    stream.Write(EncodeString(entry.Key));
                    WriteUInt32(stream, (uint)entry.Type);
                    stream.Write(entry.Bytes);
                }

                var offsets = new List<ulong>();
                ulong runningOffset = 0;
                foreach (var tensor in _tensors)
                {
                    ulong aligned = AlignUp(runningOffset, (ulong)alignment);
                    offsets.Add(aligned);
                    runningOffset = aligned + (ulong)tensor.RawBytes.Length;
                }

                for (int i = 0; i < _tensors.Count; i++)
                {
                    var tensor = _tensors[i];
                    stream.Write(EncodeString(tensor.Name));
                    WriteUInt32(stream, (uint)tensor.Shape.Length);

                    for (int d = tensor.Shape.Length - 1; d >= 0; d--)
                        WriteUInt64(stream, (ulong)tensor.Shape[d]);

                    WriteUInt32(stream, (uint)tensor.Type);
                    WriteUInt64(stream, offsets[i]);
                }

                long headerEnd = stream.Position;
                long dataPadding = (int)AlignUp((ulong)headerEnd, (ulong)alignment) - headerEnd;
                stream.Write(new byte[dataPadding]);

                long dataStart = stream.Position;
                for (int i = 0; i < _tensors.Count; i++)
                {
                    long targetPosition = dataStart + (long)offsets[i];
                    long padding = targetPosition - stream.Position;
                    if (padding > 0)
                        stream.Write(new byte[padding]);

                    stream.Write(_tensors[i].RawBytes);
                }

                return stream.ToArray();
            }

            private static ulong AlignUp(ulong value, ulong alignment)
            {
                ulong remainder = value % alignment;
                return remainder == 0 ? value : value + (alignment - remainder);
            }

            private static byte[] EncodeString(string value)
            {
                var textBytes = System.Text.Encoding.UTF8.GetBytes(value);
                using var stream = new MemoryStream();
                WriteUInt64(stream, (ulong)textBytes.Length);
                stream.Write(textBytes);
                return stream.ToArray();
            }

            private static void WriteUInt32(Stream stream, uint value) => stream.Write(BitConverter.GetBytes(value));
            private static void WriteUInt64(Stream stream, ulong value) => stream.Write(BitConverter.GetBytes(value));
        }

        private static byte[] EncodeF32(float[] values)
        {
            var bytes = new byte[values.Length * 4];
            for (int i = 0; i < values.Length; i++)
                BitConverter.GetBytes(values[i]).CopyTo(bytes, i * 4);
            return bytes;
        }

        private static byte[] EncodeF16(float[] values)
        {
            var bytes = new byte[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
            {
                ushort bits = BitConverter.HalfToUInt16Bits((Half)values[i]);
                BitConverter.GetBytes(bits).CopyTo(bytes, i * 2);
            }
            return bytes;
        }

        private static GgufFile ReadFrom(byte[] buffer) => GgufReader.Read(buffer);

        [Fact]
        public void Read_InvalidMagic_Throws()
        {
            var bytes = new GgufBuilder().WithBadMagic().Build();
            Assert.Throws<InvalidDataException>(() => ReadFrom(bytes));
        }

        [Fact]
        public void Read_UnsupportedVersion_Throws()
        {
            var bytes = new GgufBuilder().WithVersion(1).Build();
            Assert.Throws<NotSupportedException>(() => ReadFrom(bytes));
        }

        [Fact]
        public void Read_ScalarMetadataOfEveryType_RoundTrips()
        {
            var bytes = new GgufBuilder()
                .AddString("general.architecture", "llama")
                .AddUInt8("u8", 200)
                .AddInt8("i8", -5)
                .AddUInt16("u16", 4000)
                .AddInt16("i16", -300)
                .AddUInt32("u32", 70000)
                .AddInt32("i32", -70000)
                .AddUInt64("u64", 5_000_000_000UL)
                .AddInt64("i64", -5_000_000_000L)
                .AddFloat32("f32", 3.5f)
                .AddFloat64("f64", 2.25)
                .AddBool("flag", true)
                .Build();

            var file = ReadFrom(bytes);

            Assert.True(file.Metadata["general.architecture"].TryGetString(out var arch));
            Assert.Equal("llama", arch);

            Assert.True(file.Metadata["u8"].TryGetUInt64(out var u8));
            Assert.Equal(200UL, u8);

            Assert.True(file.Metadata["u16"].TryGetUInt64(out var u16));
            Assert.Equal(4000UL, u16);

            Assert.True(file.Metadata["u32"].TryGetUInt64(out var u32));
            Assert.Equal(70000UL, u32);

            Assert.True(file.Metadata["u64"].TryGetUInt64(out var u64));
            Assert.Equal(5_000_000_000UL, u64);

            Assert.True(file.Metadata["f32"].TryGetDouble(out var f32));
            Assert.Equal(3.5, f32, 5);

            Assert.True(file.Metadata["f64"].TryGetDouble(out var f64));
            Assert.Equal(2.25, f64, 10);

            Assert.Equal(GgufValueType.Bool, file.Metadata["flag"].Type);
            Assert.Equal(true, file.Metadata["flag"].Raw);
        }

        [Fact]
        public void Read_StringArrayMetadata_RoundTrips()
        {
            var bytes = new GgufBuilder()
                .AddStringArray("tokenizer.ggml.tokens", "a", "b", "c")
                .Build();

            var file = ReadFrom(bytes);

            Assert.True(file.Metadata["tokenizer.ggml.tokens"].TryGetArray(out var array));
            Assert.Equal(3, array.Count);
            Assert.True(array[1].TryGetString(out var second));
            Assert.Equal("b", second);
        }

        [Fact]
        public void Read_F32Tensor_DecodesExactValuesAndReversedShape()
        {
            var values = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.F32, new[] { 2, 3 }, EncodeF32(values))
                .Build();

            var file = ReadFrom(bytes);
            var tensor = file.Find("weight");

            Assert.NotNull(tensor);
            Assert.Equal(new[] { 2, 3 }, tensor!.Shape);
            Assert.Equal(values, tensor.Data);
        }

        [Fact]
        public void Read_F16Tensor_DecodesWithinTolerance()
        {
            var values = new float[] { -1.5f, 0f, 0.25f, 8f };
            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.F16, new[] { 4 }, EncodeF16(values))
                .Build();

            var file = ReadFrom(bytes);
            var tensor = file.Find("weight")!;

            for (int i = 0; i < values.Length; i++)
                Assert.Equal(values[i], tensor.Data[i], 3);
        }

        [Fact]
        public void Read_Q8_0Tensor_DecodesUsingBlockScale()
        {
            const float scale = 0.5f;
            var quantized = new sbyte[32];
            var expected = new float[32];
            for (int i = 0; i < 32; i++)
            {
                quantized[i] = (sbyte)(i - 16);
                expected[i] = quantized[i] * scale;
            }

            using var rawStream = new MemoryStream();
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)scale)));
            foreach (var q in quantized)
                rawStream.WriteByte(unchecked((byte)q));

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q8_0, new[] { 32 }, rawStream.ToArray())
                .Build();

            var file = ReadFrom(bytes);
            var tensor = file.Find("weight")!;

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], tensor.Data[i], 4);
        }

        [Fact]
        public void Read_Q4_0Tensor_DecodesUsingPackedNibbles()
        {
            const float scale = 0.25f;
            var packed = new byte[16];
            var expected = new float[32];

            for (int j = 0; j < 16; j++)
            {
                packed[j] = (byte)(j | ((15 - j) << 4));
                expected[j] = (j - 8) * scale;
                expected[j + 16] = ((15 - j) - 8) * scale;
            }

            using var rawStream = new MemoryStream();
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)scale)));
            rawStream.Write(packed);

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q4_0, new[] { 32 }, rawStream.ToArray())
                .Build();

            var file = ReadFrom(bytes);
            var tensor = file.Find("weight")!;

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], tensor.Data[i], 5);
        }

        [Fact]
        public void Read_Q4_1Tensor_DecodesScaleAndMinimum()
        {
            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q4_1, new[] { 32 }, new byte[20])
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;
            Assert.Equal(32, tensor.Data.Length);
            Assert.All(tensor.Data, value => Assert.Equal(0f, value));
        }

        [Fact]
        public void Read_Q8_1Tensor_UsesHalfScaleAndPackedSignedValues()
        {
            using var rawStream = new MemoryStream();
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)0.5f)));
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)12f)));
            for (int i = 0; i < 32; i++)
                rawStream.WriteByte(unchecked((byte)(sbyte)(i - 16)));

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q8_1, new[] { 32 }, rawStream.ToArray())
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;
            for (int i = 0; i < 32; i++)
                Assert.Equal((i - 16) * 0.5f, tensor.Data[i]);
        }

        [Fact]
        public void Read_Q5_0Tensor_DecodesHighBitsAndSignedOffset()
        {
            using var rawStream = new MemoryStream();
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)0.5f)));
            rawStream.Write(BitConverter.GetBytes(0x00010001u));
            rawStream.Write(new byte[16]);

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q5_0, new[] { 32 }, rawStream.ToArray())
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;
            Assert.Equal(0f, tensor.Data[0]);
            Assert.Equal(0f, tensor.Data[16]);
            for (int i = 1; i < 16; i++)
                Assert.Equal(-8f, tensor.Data[i]);
            for (int i = 17; i < 32; i++)
                Assert.Equal(-8f, tensor.Data[i]);
        }

        [Fact]
        public void Read_Q2_KTensor_DecodesUsingPackedScalesAndMins()
        {
            const float d = 2f;
            const float dmin = 1f;

            var scales = new byte[16];
            for (int i = 0; i < scales.Length; i++)
                scales[i] = 0x21;

            var qs = new byte[64];
            for (int i = 0; i < qs.Length; i++)
                qs[i] = 0xFF;

            using var rawStream = new MemoryStream();
            rawStream.Write(scales);
            rawStream.Write(qs);
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)d)));
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)dmin)));

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q2_K, new[] { 256 }, rawStream.ToArray())
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;

            for (int i = 0; i < tensor.Data.Length; i++)
                Assert.Equal(4f, tensor.Data[i], 4);
        }

        [Fact]
        public void Read_Q3_KTensor_DecodesUsingHighBitAndPackedScales()
        {
            const float d = 1f;
            var hmask = new byte[32];
            var qs = new byte[64];
            for (int i = 0; i < qs.Length; i++)
                qs[i] = 0xFF;

            var scales = new byte[]
            {
                0x11, 0x11, 0x11, 0x11,
                0x11, 0x11, 0x11, 0x11,
                0xAA, 0xAA, 0xAA, 0xAA
            };

            using var rawStream = new MemoryStream();
            rawStream.Write(hmask);
            rawStream.Write(qs);
            rawStream.Write(scales);
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)d)));

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q3_K, new[] { 256 }, rawStream.ToArray())
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;

            for (int i = 0; i < tensor.Data.Length; i++)
                Assert.Equal(-1f, tensor.Data[i], 4);
        }

        [Fact]
        public void Read_Q4_KTensor_DecodesUsingPackedScalesAndMins()
        {
            const float d = 2f;
            const float dmin = 5f;
            var scales = new byte[] { 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };

            var qs = new byte[128];
            for (int i = 0; i < qs.Length; i++)
                qs[i] = 0x21;

            using var rawStream = new MemoryStream();
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)d)));
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)dmin)));
            rawStream.Write(scales);
            rawStream.Write(qs);

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q4_K, new[] { 256 }, rawStream.ToArray())
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;

            var expected = new float[256];
            for (int chunk = 0; chunk < 4; chunk++)
            {
                int baseIdx = chunk * 64;
                for (int l = 0; l < 32; l++)
                {
                    expected[baseIdx + l] = 2f;
                    expected[baseIdx + 32 + l] = 4f;
                }
            }

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], tensor.Data[i], 4);
        }

        [Fact]
        public void Read_Q5_KTensor_DecodesIncludingHighBit()
        {
            const float d = 2f;
            const float dmin = 0f;
            var scales = new byte[] { 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 };

            var ql = new byte[128];
            for (int i = 0; i < ql.Length; i++)
                ql[i] = 0x21;

            var qh = new byte[32];
            for (int i = 0; i < qh.Length; i++)
                qh[i] = 0xFF;

            using var rawStream = new MemoryStream();
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)d)));
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)dmin)));
            rawStream.Write(scales);
            rawStream.Write(qh);
            rawStream.Write(ql);

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q5_K, new[] { 256 }, rawStream.ToArray())
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;

            var expected = new float[256];
            for (int chunk = 0; chunk < 4; chunk++)
            {
                int baseIdx = chunk * 64;
                for (int l = 0; l < 32; l++)
                {
                    expected[baseIdx + l] = 34f;
                    expected[baseIdx + 32 + l] = 36f;
                }
            }

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], tensor.Data[i], 4);
        }

        [Fact]
        public void Read_Q6_KTensor_DecodesSignedQuantsWithScale()
        {
            const float d = 1f;

            var ql = new byte[128];
            for (int i = 0; i < 32; i++) { ql[i] = 0x00; ql[32 + i] = 0xFF; }
            for (int i = 0; i < 32; i++) { ql[64 + i] = 0x00; ql[96 + i] = 0xFF; }

            var qh = new byte[64];
            var scales = new byte[16];
            for (int i = 0; i < scales.Length; i++)
                scales[i] = 1;

            using var rawStream = new MemoryStream();
            rawStream.Write(ql);
            rawStream.Write(qh);
            rawStream.Write(scales);
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)d)));

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q6_K, new[] { 256 }, rawStream.ToArray())
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;

            var expected = new float[256];
            for (int chunk = 0; chunk < 2; chunk++)
            {
                int baseIdx = chunk * 128;
                for (int l = 0; l < 32; l++)
                {
                    expected[baseIdx + l] = -32f;
                    expected[baseIdx + 32 + l] = -17f;
                    expected[baseIdx + 64 + l] = -32f;
                    expected[baseIdx + 96 + l] = -17f;
                }
            }

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], tensor.Data[i], 4);
        }

        [Fact]
        public void Read_Q8_KTensor_DecodesLinearScale()
        {
            const float d = 0.5f;
            var qs = new sbyte[256];
            var expected = new float[256];
            for (int i = 0; i < 256; i++)
            {
                qs[i] = (sbyte)(i - 128);
                expected[i] = qs[i] * d;
            }

            using var rawStream = new MemoryStream();
            rawStream.Write(BitConverter.GetBytes(d));
            foreach (var q in qs)
                rawStream.WriteByte(unchecked((byte)q));
            rawStream.Write(new byte[32]);

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.Q8_K, new[] { 256 }, rawStream.ToArray())
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], tensor.Data[i], 5);
        }

        [Fact]
        public void Read_Iq4NlTensor_DecodesUsingNonLinearTable()
        {
            const float d = 1.5f;
            var kvalues = new sbyte[] { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

            var qs = new byte[16];
            var expected = new float[32];
            for (int j = 0; j < 16; j++)
            {
                qs[j] = (byte)(j | ((15 - j) << 4));
                expected[j] = d * kvalues[j];
                expected[j + 16] = d * kvalues[15 - j];
            }

            using var rawStream = new MemoryStream();
            rawStream.Write(BitConverter.GetBytes(BitConverter.HalfToUInt16Bits((Half)d)));
            rawStream.Write(qs);

            var bytes = new GgufBuilder()
                .AddTensor("weight", GgmlType.IQ4_NL, new[] { 32 }, rawStream.ToArray())
                .Build();

            var tensor = ReadFrom(bytes).Find("weight")!;

            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], tensor.Data[i], 4);
        }

        [Fact]
        public void Read_MultipleTensorsWithCustomAlignment_RespectsPadding()
        {
            var first = new float[] { 1f, 2f };
            var second = new float[] { 3f, 4f, 5f };

            var bytes = new GgufBuilder()
                .AddUInt32("general.alignment", 64)
                .AddTensor("first", GgmlType.F32, new[] { 2 }, EncodeF32(first))
                .AddTensor("second", GgmlType.F32, new[] { 3 }, EncodeF32(second))
                .Build(alignment: 64);

            var file = ReadFrom(bytes);

            Assert.Equal(first, file.Find("first")!.Data);
            Assert.Equal(second, file.Find("second")!.Data);
        }
    }
}
