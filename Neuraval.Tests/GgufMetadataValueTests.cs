using Neuraval.Core.Serialization.Gguf;
using Xunit;

namespace Neuraval.Tests
{
    public class GgufMetadataValueTests
    {
        [Fact]
        public void TryGetUInt64_FromUInt32_ReturnsTrue()
        {
            var value = new GgufMetadataValue(GgufValueType.UInt32, (uint)42);
            Assert.True(value.TryGetUInt64(out var result));
            Assert.Equal(42UL, result);
        }

        [Fact]
        public void TryGetUInt64_FromNegativeInt32_ReturnsFalse()
        {
            var value = new GgufMetadataValue(GgufValueType.Int32, -1);
            Assert.False(value.TryGetUInt64(out _));
        }

        [Fact]
        public void TryGetUInt64_FromString_ReturnsFalse()
        {
            var value = new GgufMetadataValue(GgufValueType.String, "hola");
            Assert.False(value.TryGetUInt64(out _));
        }

        [Fact]
        public void TryGetDouble_FromFloat32_ReturnsTrue()
        {
            var value = new GgufMetadataValue(GgufValueType.Float32, 1.5f);
            Assert.True(value.TryGetDouble(out var result));
            Assert.Equal(1.5, result, 6);
        }

        [Fact]
        public void TryGetDouble_FromIntegerRaw_CoercesToDouble()
        {
            var value = new GgufMetadataValue(GgufValueType.UInt32, (uint)8);
            Assert.True(value.TryGetDouble(out var result));
            Assert.Equal(8.0, result, 6);
        }

        [Fact]
        public void TryGetString_FromNonString_ReturnsFalse()
        {
            var value = new GgufMetadataValue(GgufValueType.UInt32, (uint)1);
            Assert.False(value.TryGetString(out _));
        }

        [Fact]
        public void TryGetArray_FromArray_ReturnsElements()
        {
            var elements = new[]
            {
                new GgufMetadataValue(GgufValueType.String, "a"),
                new GgufMetadataValue(GgufValueType.String, "b")
            };
            var value = new GgufMetadataValue(GgufValueType.Array, elements);

            Assert.True(value.TryGetArray(out var result));
            Assert.Equal(2, result.Count);
        }
    }
}
