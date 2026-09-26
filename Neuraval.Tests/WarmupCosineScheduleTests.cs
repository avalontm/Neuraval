using System;
using Neuraval.Core.Training.Scheduling;
using Xunit;

namespace Neuraval.Tests
{
    public class WarmupCosineScheduleTests
    {
        [Fact]
        public void Constructor_NullOptions_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new WarmupCosineSchedule(null!));
        }

        [Fact]
        public void LearningRateAt_NegativeGlobalStep_Throws()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(0.001f, 100, 1000, 0f));

            Assert.Throws<ArgumentOutOfRangeException>(() => schedule.LearningRateAt(-1));
        }

        [Fact]
        public void LearningRateAt_StepZero_StartsAtZeroDuringWarmup()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 4, 10, 0.0f));

            Assert.Equal(0.0f, schedule.LearningRateAt(0), 5);
        }

        [Fact]
        public void LearningRateAt_MidWarmup_IsLinear()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 4, 10, 0.0f));

            Assert.Equal(0.5f, schedule.LearningRateAt(2), 5);
        }

        [Fact]
        public void LearningRateAt_WarmupEnd_ReachesPeak()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 4, 10, 0.0f));

            Assert.Equal(1.0f, schedule.LearningRateAt(4), 5);
        }

        [Fact]
        public void LearningRateAt_MidDecay_MatchesCosineFormula()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 4, 10, 0.0f));

            Assert.Equal(0.5f, schedule.LearningRateAt(7), 5);
        }

        [Fact]
        public void LearningRateAt_AtMaxSteps_EqualsMinLearningRate()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 4, 10, 0.0f));

            Assert.Equal(0.0f, schedule.LearningRateAt(10), 5);
        }

        [Fact]
        public void LearningRateAt_BeyondMaxSteps_StaysAtMinLearningRate()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 4, 10, 0.0f));

            Assert.Equal(0.0f, schedule.LearningRateAt(1000), 5);
        }

        [Fact]
        public void LearningRateAt_NonZeroMinLearningRate_FloorsDecay()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 4, 10, 0.2f));

            Assert.Equal(0.2f, schedule.LearningRateAt(10), 5);
        }

        [Fact]
        public void LearningRateAt_ZeroWarmupSteps_StartsAtPeak()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 0, 4, 0.2f));

            Assert.Equal(1.0f, schedule.LearningRateAt(0), 5);
        }

        [Fact]
        public void LearningRateAt_ZeroWarmupMidDecay_MatchesCosineFormula()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 0, 4, 0.2f));

            Assert.Equal(0.6f, schedule.LearningRateAt(2), 5);
        }

        [Fact]
        public void LearningRateAt_IsMonotonicallyNonIncreasingAfterWarmup()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 4, 100, 0.05f));

            float previous = schedule.LearningRateAt(4);

            for (int step = 5; step <= 100; step++)
            {
                float current = schedule.LearningRateAt(step);
                Assert.True(current <= previous + 1e-6f);
                previous = current;
            }
        }

        [Fact]
        public void AsFunc_DelegatesToLearningRateAt()
        {
            var schedule = new WarmupCosineSchedule(WarmupCosineScheduleOptions.Create(1.0f, 4, 10, 0.0f));
            var func = schedule.AsFunc();

            Assert.Equal(schedule.LearningRateAt(7), func(7));
        }
    }
}
