using System;
using Neuraval.Core.Training.Scheduling;
using Xunit;

namespace Neuraval.Tests
{
    public class ConstantLearningRateScheduleTests
    {
        [Fact]
        public void LearningRateAt_AnyStep_ReturnsSameValue()
        {
            var schedule = new ConstantLearningRateSchedule(0.001f);

            Assert.Equal(0.001f, schedule.LearningRateAt(0));
            Assert.Equal(0.001f, schedule.LearningRateAt(1000));
            Assert.Equal(0.001f, schedule.LearningRateAt(999999));
        }

        [Fact]
        public void Constructor_NegativeLearningRate_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new ConstantLearningRateSchedule(-0.001f));
        }

        [Fact]
        public void LearningRateAt_NegativeGlobalStep_Throws()
        {
            var schedule = new ConstantLearningRateSchedule(0.001f);

            Assert.Throws<ArgumentOutOfRangeException>(() => schedule.LearningRateAt(-1));
        }

        [Fact]
        public void AsFunc_DelegatesToLearningRateAt()
        {
            var schedule = new ConstantLearningRateSchedule(0.02f);
            var func = schedule.AsFunc();

            Assert.Equal(schedule.LearningRateAt(5), func(5));
        }
    }
}
