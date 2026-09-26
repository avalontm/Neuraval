using System;
using Neuraval.Core.Training.Stability;
using Xunit;

namespace Neuraval.Tests
{
    public class LossDivergenceOptionsTests
    {
        [Fact]
        public void Create_Defaults_MatchExpectedValues()
        {
            var options = LossDivergenceOptions.Create();

            Assert.Equal(2.0f, options.DivergenceFactor);
            Assert.Equal(10, options.WarmupSteps);
        }

        [Fact]
        public void Create_CustomValues_ExposesThem()
        {
            var options = LossDivergenceOptions.Create(1.5f, 5);

            Assert.Equal(1.5f, options.DivergenceFactor);
            Assert.Equal(5, options.WarmupSteps);
        }

        [Theory]
        [InlineData(1f)]
        [InlineData(0.5f)]
        public void Create_DivergenceFactorNotAboveOne_Throws(float divergenceFactor)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => LossDivergenceOptions.Create(divergenceFactor: divergenceFactor));
        }

        [Fact]
        public void Create_NegativeWarmupSteps_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => LossDivergenceOptions.Create(warmupSteps: -1));
        }
    }
}
