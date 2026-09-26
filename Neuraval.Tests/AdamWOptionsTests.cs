using System;
using Neuraval.Core.Training.Optimization;
using Xunit;

namespace Neuraval.Tests
{
    public class AdamWOptionsTests
    {
        [Fact]
        public void Create_Defaults_MatchRoadmapValues()
        {
            var options = AdamWOptions.Create();

            Assert.Equal(0.0003f, options.LearningRate);
            Assert.Equal(0.1f, options.WeightDecay);
            Assert.Equal(0.9f, options.Beta1);
            Assert.Equal(0.95f, options.Beta2);
            Assert.Equal(1e-8f, options.Epsilon);
        }

        [Fact]
        public void Create_CustomValues_ExposesThem()
        {
            var options = AdamWOptions.Create(0.001f, 0.05f, 0.85f, 0.999f, 1e-6f);

            Assert.Equal(0.001f, options.LearningRate);
            Assert.Equal(0.05f, options.WeightDecay);
            Assert.Equal(0.85f, options.Beta1);
            Assert.Equal(0.999f, options.Beta2);
            Assert.Equal(1e-6f, options.Epsilon);
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-0.001f)]
        public void Create_LearningRateNotPositive_Throws(float learningRate)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => AdamWOptions.Create(learningRate: learningRate));
        }

        [Fact]
        public void Create_WeightDecayNegative_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => AdamWOptions.Create(weightDecay: -0.1f));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(1f)]
        public void Create_Beta1OutOfRange_Throws(float beta1)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => AdamWOptions.Create(beta1: beta1));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(1f)]
        public void Create_Beta2OutOfRange_Throws(float beta2)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => AdamWOptions.Create(beta2: beta2));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-1e-8f)]
        public void Create_EpsilonNotPositive_Throws(float epsilon)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => AdamWOptions.Create(epsilon: epsilon));
        }
    }
}
