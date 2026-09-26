using System;
using Neuraval.Core.Training.Loop;
using Xunit;

namespace Neuraval.Tests
{
    public class TrainingStepOutcomeTests
    {
        [Fact]
        public void Constructor_ExposesAllFields()
        {
            var outcome = new TrainingStepOutcome(2.5f, 0.8f, 512, 4, 1.25);

            Assert.Equal(2.5f, outcome.Loss);
            Assert.Equal(0.8f, outcome.GradientNorm);
            Assert.Equal(512, outcome.TokensProcessed);
            Assert.Equal(4, outcome.SamplesProcessed);
            Assert.Equal(1.25, outcome.ElapsedSeconds);
        }

        [Fact]
        public void Constructor_NegativeLoss_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingStepOutcome(-0.1f, 0.5f, 10, 1, 1.0));
        }

        [Fact]
        public void Constructor_TokensProcessedNotPositive_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingStepOutcome(1.0f, 0.5f, 0, 1, 1.0));
        }

        [Fact]
        public void Constructor_SamplesProcessedNotPositive_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingStepOutcome(1.0f, 0.5f, 10, 0, 1.0));
        }

        [Fact]
        public void Constructor_ElapsedSecondsNotPositive_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingStepOutcome(1.0f, 0.5f, 10, 1, 0.0));
        }
    }
}
