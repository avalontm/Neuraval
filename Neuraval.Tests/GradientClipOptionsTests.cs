using System;
using Neuraval.Core.Training.Stability;
using Xunit;

namespace Neuraval.Tests
{
    public class GradientClipOptionsTests
    {
        [Fact]
        public void Create_ValidMaxNorm_ExposesIt()
        {
            var options = GradientClipOptions.Create(1.0f);

            Assert.Equal(1.0f, options.MaxNorm);
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-1f)]
        public void Create_MaxNormNotPositive_Throws(float maxNorm)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => GradientClipOptions.Create(maxNorm));
        }
    }
}
