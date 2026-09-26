using System;
using Neuraval.Core.Training.Loop;
using Xunit;

namespace Neuraval.Tests
{
    public class TrainingStepMetricsTests
    {
        [Fact]
        public void Create_ZeroLoss_PerplexityIsOne()
        {
            var metrics = TrainingStepMetrics.Create(1, 100, 0f, 0.0003f, 0.5f, 1000f, 8f);

            Assert.Equal(1f, metrics.Perplexity, 4);
        }

        [Fact]
        public void Create_KnownLoss_ComputesExpectedPerplexity()
        {
            float loss = MathF.Log(2f);

            var metrics = TrainingStepMetrics.Create(1, 100, loss, 0.0003f, 0.5f, 1000f, 8f);

            Assert.Equal(2f, metrics.Perplexity, 3);
        }

        [Fact]
        public void Create_ExposesAllFields()
        {
            var metrics = TrainingStepMetrics.Create(1200, 18432000, 2.481f, 0.000241f, 0.82f, 41200f, 32f, 4096L);

            Assert.Equal(1200, metrics.GlobalStep);
            Assert.Equal(18432000, metrics.TokensSeen);
            Assert.Equal(2.481f, metrics.Loss);
            Assert.Equal(0.000241f, metrics.LearningRate);
            Assert.Equal(0.82f, metrics.GradientNorm);
            Assert.Equal(41200f, metrics.TokensPerSecond);
            Assert.Equal(32f, metrics.SamplesPerSecond);
            Assert.Equal(4096L, metrics.GpuMemoryBytes);
        }

        [Fact]
        public void Create_GpuMemoryBytesOmitted_IsNull()
        {
            var metrics = TrainingStepMetrics.Create(1, 100, 1f, 0.0003f, 0.5f, 1000f, 8f);

            Assert.Null(metrics.GpuMemoryBytes);
        }

        [Fact]
        public void Create_NegativeGlobalStep_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => TrainingStepMetrics.Create(-1, 0, 1f, 0.0003f, 0.5f, 100f, 1f));
        }

        [Fact]
        public void Create_NegativeTokensSeen_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => TrainingStepMetrics.Create(1, -1, 1f, 0.0003f, 0.5f, 100f, 1f));
        }

        [Fact]
        public void Create_NegativeLoss_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => TrainingStepMetrics.Create(1, 0, -1f, 0.0003f, 0.5f, 100f, 1f));
        }

        [Fact]
        public void ToLogString_ContainsAllRegisteredFields()
        {
            var metrics = TrainingStepMetrics.Create(1200, 18432000, 2.481f, 0.000241f, 0.82f, 41200f, 32f);

            var log = metrics.ToLogString();

            Assert.Contains("Step: 1200", log);
            Assert.Contains("Tokens: 18,432,000", log);
            Assert.Contains("Loss: 2.481", log);
            Assert.Contains("LR: 0.000241", log);
            Assert.Contains("Grad norm: 0.82", log);
            Assert.Contains("Tokens/sec: 41,200", log);
        }
    }
}
