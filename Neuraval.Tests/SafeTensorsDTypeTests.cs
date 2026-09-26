using System;
using Neuraval.Core.Serialization.SafeTensors;
using Xunit;

namespace Neuraval.Tests
{
    public class SafeTensorsDTypeTests
    {
        [Theory]
        [InlineData(SafeTensorsDType.F32, "F32")]
        [InlineData(SafeTensorsDType.F16, "F16")]
        [InlineData(SafeTensorsDType.BF16, "BF16")]
        public void ToTag_ReturnsExpectedTag(SafeTensorsDType dtype, string expectedTag)
        {
            Assert.Equal(expectedTag, dtype.ToTag());
        }

        [Theory]
        [InlineData("F32", SafeTensorsDType.F32)]
        [InlineData("F16", SafeTensorsDType.F16)]
        [InlineData("BF16", SafeTensorsDType.BF16)]
        public void FromTag_ReturnsExpectedDType(string tag, SafeTensorsDType expected)
        {
            Assert.Equal(expected, SafeTensorsDTypeExtensions.FromTag(tag));
        }

        [Fact]
        public void FromTag_UnknownTag_Throws()
        {
            Assert.Throws<NotSupportedException>(() => SafeTensorsDTypeExtensions.FromTag("INT8"));
        }

        [Theory]
        [InlineData(SafeTensorsDType.F32, 4)]
        [InlineData(SafeTensorsDType.F16, 2)]
        [InlineData(SafeTensorsDType.BF16, 2)]
        public void ByteSize_ReturnsExpectedSize(SafeTensorsDType dtype, int expectedSize)
        {
            Assert.Equal(expectedSize, dtype.ByteSize());
        }
    }
}
