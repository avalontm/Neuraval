using System;
using Neuraval.Core.Training.Optimization;
using Xunit;

namespace Neuraval.Tests
{
    public class AdamWVectorOptimizerTests
    {
        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        public void Constructor_LengthNotPositive_Throws(int length)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new AdamWVectorOptimizer(length));
        }

        [Fact]
        public void Constructor_NoOptions_UsesDefaultOptions()
        {
            var optimizer = new AdamWVectorOptimizer(3);

            Assert.Equal(0.0003f, optimizer.Options.LearningRate);
            Assert.Equal(0, optimizer.TimeStep);
        }

        [Fact]
        public void Update_SingleStepNoDecay_MatchesAdamFormula()
        {
            var options = AdamWOptions.Create(learningRate: 0.1f, weightDecay: 0f, beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f);
            var optimizer = new AdamWVectorOptimizer(1, options);

            var parameters = new[] { 1.0f };
            var gradients = new[] { 0.5f };

            optimizer.Update(parameters, gradients);

            Assert.Equal(0.9f, parameters[0], 5);
        }

        [Fact]
        public void Update_ZeroGradientWithWeightDecay_ShrinksParameterTowardZero()
        {
            var options = AdamWOptions.Create(learningRate: 0.1f, weightDecay: 0.1f);
            var optimizer = new AdamWVectorOptimizer(1, options);

            var parameters = new[] { 2.0f };
            var gradients = new[] { 0.0f };

            optimizer.Update(parameters, gradients);

            Assert.Equal(1.98f, parameters[0], 5);
        }

        [Fact]
        public void Update_ParameterLengthMismatch_Throws()
        {
            var optimizer = new AdamWVectorOptimizer(2);

            Assert.Throws<ArgumentException>(() => optimizer.Update(new float[3], new float[2]));
        }

        [Fact]
        public void Update_GradientLengthMismatch_Throws()
        {
            var optimizer = new AdamWVectorOptimizer(2);

            Assert.Throws<ArgumentException>(() => optimizer.Update(new float[2], new float[3]));
        }

        [Fact]
        public void Update_IncrementsTimeStep()
        {
            var optimizer = new AdamWVectorOptimizer(1);
            var parameters = new[] { 1.0f };
            var gradients = new[] { 0.1f };

            optimizer.Update(parameters, gradients);
            optimizer.Update(parameters, gradients);

            Assert.Equal(2, optimizer.TimeStep);
        }

        [Fact]
        public void Update_WithoutLearningRateOverride_UsesOptionsLearningRate()
        {
            var options = AdamWOptions.Create(learningRate: 0.1f, weightDecay: 0f, beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f);
            var optimizerA = new AdamWVectorOptimizer(1, options);
            var optimizerB = new AdamWVectorOptimizer(1, options);

            var parametersA = new[] { 1.0f };
            var parametersB = new[] { 1.0f };

            optimizerA.Update(parametersA, new[] { 0.5f });
            optimizerB.Update(parametersB, new[] { 0.5f }, options.LearningRate);

            Assert.Equal(parametersB[0], parametersA[0], 6);
        }

        [Fact]
        public void Reset_ClearsStateAndTimeStep()
        {
            var optimizer = new AdamWVectorOptimizer(1);
            var warmup = new[] { 1.0f };

            optimizer.Update(warmup, new[] { 0.5f });
            optimizer.Reset();

            Assert.Equal(0, optimizer.TimeStep);

            var parameters = new[] { 1.0f };
            optimizer.Update(parameters, new[] { 0.5f });

            var freshOptimizer = new AdamWVectorOptimizer(1, optimizer.Options);
            var freshParameters = new[] { 1.0f };
            freshOptimizer.Update(freshParameters, new[] { 0.5f });

            Assert.Equal(freshParameters[0], parameters[0], 6);
        }

        [Fact]
        public void SaveState_LoadStateInto_RoundTripPreservesBehavior()
        {
            var optimizer = new AdamWVectorOptimizer(2);
            optimizer.Update(new[] { 1.0f, 2.0f }, new[] { 0.3f, -0.2f });

            var state = optimizer.SaveState();

            var restored = new AdamWVectorOptimizer(2, optimizer.Options);
            restored.LoadStateInto(state);

            var parametersOriginal = new[] { 5.0f, 5.0f };
            var parametersRestored = new[] { 5.0f, 5.0f };
            var gradients = new[] { 0.1f, 0.1f };

            optimizer.Update(parametersOriginal, gradients);
            restored.Update(parametersRestored, gradients);

            Assert.Equal(parametersOriginal[0], parametersRestored[0], 6);
            Assert.Equal(parametersOriginal[1], parametersRestored[1], 6);
            Assert.Equal(optimizer.TimeStep, restored.TimeStep);
        }

        [Fact]
        public void LoadStateInto_LengthMismatch_Throws()
        {
            var optimizer = new AdamWVectorOptimizer(2);
            var state = new AdamWVectorOptimizerState { Length = 3, M = new float[3], V = new float[3], TimeStep = 0 };

            Assert.Throws<ArgumentException>(() => optimizer.LoadStateInto(state));
        }
    }
}
