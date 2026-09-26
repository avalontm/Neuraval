using System;
using Neuraval.Core.Training.Optimization;
using Xunit;

namespace Neuraval.Tests
{
    public class AdamWMatrixOptimizerTests
    {
        [Theory]
        [InlineData(0, 2)]
        [InlineData(-1, 2)]
        public void Constructor_RowsNotPositive_Throws(int rows, int cols)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new AdamWMatrixOptimizer(rows, cols));
        }

        [Theory]
        [InlineData(2, 0)]
        [InlineData(2, -1)]
        public void Constructor_ColsNotPositive_Throws(int rows, int cols)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new AdamWMatrixOptimizer(rows, cols));
        }

        [Fact]
        public void Update_SingleStepNoDecay_MatchesVectorFormula()
        {
            var options = AdamWOptions.Create(learningRate: 0.1f, weightDecay: 0f, beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f);
            var optimizer = new AdamWMatrixOptimizer(1, 1, options);

            var parameters = new float[1, 1] { { 1.0f } };
            var gradients = new float[1, 1] { { 0.5f } };

            optimizer.Update(parameters, gradients);

            Assert.Equal(0.9f, parameters[0, 0], 5);
        }

        [Fact]
        public void Update_ParameterShapeMismatch_Throws()
        {
            var optimizer = new AdamWMatrixOptimizer(2, 2);

            Assert.Throws<ArgumentException>(() => optimizer.Update(new float[3, 2], new float[2, 2]));
        }

        [Fact]
        public void Update_GradientShapeMismatch_Throws()
        {
            var optimizer = new AdamWMatrixOptimizer(2, 2);

            Assert.Throws<ArgumentException>(() => optimizer.Update(new float[2, 2], new float[2, 3]));
        }

        [Fact]
        public void Update_IncrementsTimeStep()
        {
            var optimizer = new AdamWMatrixOptimizer(1, 1);
            var parameters = new float[1, 1] { { 1.0f } };
            var gradients = new float[1, 1] { { 0.1f } };

            optimizer.Update(parameters, gradients);
            optimizer.Update(parameters, gradients);

            Assert.Equal(2, optimizer.TimeStep);
        }

        [Fact]
        public void Reset_ClearsStateAndTimeStep()
        {
            var optimizer = new AdamWMatrixOptimizer(1, 1);
            optimizer.Update(new float[1, 1] { { 1.0f } }, new float[1, 1] { { 0.5f } });

            optimizer.Reset();

            Assert.Equal(0, optimizer.TimeStep);

            var parameters = new float[1, 1] { { 1.0f } };
            optimizer.Update(parameters, new float[1, 1] { { 0.5f } });

            var freshOptimizer = new AdamWMatrixOptimizer(1, 1, optimizer.Options);
            var freshParameters = new float[1, 1] { { 1.0f } };
            freshOptimizer.Update(freshParameters, new float[1, 1] { { 0.5f } });

            Assert.Equal(freshParameters[0, 0], parameters[0, 0], 6);
        }

        [Fact]
        public void SaveState_LoadStateInto_RoundTripPreservesBehavior()
        {
            var optimizer = new AdamWMatrixOptimizer(1, 2);
            optimizer.Update(new float[1, 2] { { 1.0f, 2.0f } }, new float[1, 2] { { 0.3f, -0.2f } });

            var state = optimizer.SaveState();

            var restored = new AdamWMatrixOptimizer(1, 2, optimizer.Options);
            restored.LoadStateInto(state);

            var parametersOriginal = new float[1, 2] { { 5.0f, 5.0f } };
            var parametersRestored = new float[1, 2] { { 5.0f, 5.0f } };
            var gradients = new float[1, 2] { { 0.1f, 0.1f } };

            optimizer.Update(parametersOriginal, gradients);
            restored.Update(parametersRestored, gradients);

            Assert.Equal(parametersOriginal[0, 0], parametersRestored[0, 0], 6);
            Assert.Equal(parametersOriginal[0, 1], parametersRestored[0, 1], 6);
            Assert.Equal(optimizer.TimeStep, restored.TimeStep);
        }

        [Fact]
        public void LoadStateInto_ShapeMismatch_Throws()
        {
            var optimizer = new AdamWMatrixOptimizer(2, 2);
            var state = new AdamWMatrixOptimizerState { Rows = 3, Cols = 2, M = new float[6], V = new float[6], TimeStep = 0 };

            Assert.Throws<ArgumentException>(() => optimizer.LoadStateInto(state));
        }
    }
}
