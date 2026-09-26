using System;
using Neuraval.Core.Training.Scheduling;
using Xunit;

namespace Neuraval.Tests
{
    public class WarmupCosineScheduleOptionsTests
    {
        [Fact]
        public void Create_ValidValues_ExposesThem()
        {
            var options = WarmupCosineScheduleOptions.Create(0.0003f, 500, 10000, 0.00003f);

            Assert.Equal(0.0003f, options.PeakLearningRate);
            Assert.Equal(500, options.WarmupSteps);
            Assert.Equal(10000, options.MaxSteps);
            Assert.Equal(0.00003f, options.MinLearningRate);
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-0.001f)]
        public void Create_PeakLearningRateNotPositive_Throws(float peak)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => WarmupCosineScheduleOptions.Create(peak, 100, 1000, 0f));
        }

        [Fact]
        public void Create_NegativeWarmupSteps_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => WarmupCosineScheduleOptions.Create(0.001f, -1, 1000, 0f));
        }

        [Theory]
        [InlineData(100, 100)]
        [InlineData(100, 50)]
        public void Create_MaxStepsNotGreaterThanWarmupSteps_Throws(int warmupSteps, int maxSteps)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => WarmupCosineScheduleOptions.Create(0.001f, warmupSteps, maxSteps, 0f));
        }

        [Fact]
        public void Create_NegativeMinLearningRate_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => WarmupCosineScheduleOptions.Create(0.001f, 100, 1000, -0.0001f));
        }

        [Fact]
        public void Create_MinLearningRateGreaterThanPeak_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => WarmupCosineScheduleOptions.Create(0.001f, 100, 1000, 0.01f));
        }

        [Fact]
        public void Create_ZeroWarmupSteps_IsAllowed()
        {
            var options = WarmupCosineScheduleOptions.Create(0.001f, 0, 1000, 0f);

            Assert.Equal(0, options.WarmupSteps);
        }
    }
}
