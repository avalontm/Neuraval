using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class RMSNormTests
    {
        [Fact]
        public void Forward_WithKnownVector_MatchesManualComputation()
        {
            var norm = new RMSNorm(4, epsilon: 1e-6f);
            var input = new float[,] { { 1f, 2f, 3f, 4f } };

            var output = norm.Forward(input);

            float meanSquares = (1f + 4f + 9f + 16f) / 4f;
            float rms = MathF.Sqrt(meanSquares + 1e-6f);

            Assert.Equal(1f / rms, output[0, 0], precision: 5);
            Assert.Equal(2f / rms, output[0, 1], precision: 5);
            Assert.Equal(3f / rms, output[0, 2], precision: 5);
            Assert.Equal(4f / rms, output[0, 3], precision: 5);
        }

        [Fact]
        public void Forward_DoesNotUseMeanSubtraction()
        {
            var norm = new RMSNorm(3, epsilon: 1e-6f);
            var input = new float[,] { { 5f, 5f, 5f } };

            var output = norm.Forward(input);

            Assert.Equal(1f, output[0, 0], precision: 4);
            Assert.Equal(1f, output[0, 1], precision: 4);
            Assert.Equal(1f, output[0, 2], precision: 4);
        }

        [Fact]
        public void Backward_InputGradient_MatchesNumericalGradient()
        {
            var rows = 2;
            var dim = 5;
            var input = new float[,]
            {
                { 0.5f, -1.2f, 2.3f, 0.1f, -0.7f },
                { 1.1f, 0.3f, -2.0f, 0.4f, 1.8f }
            };
            var upstream = new float[,]
            {
                { 0.3f, -0.1f, 0.7f, 0.2f, -0.4f },
                { -0.2f, 0.5f, 0.1f, -0.6f, 0.3f }
            };

            var norm = new RMSNorm(dim, epsilon: 1e-6f);
            norm.Forward(input);
            var analyticalGrad = norm.Backward(upstream);

            float epsilon = 1e-3f;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    var plus = (float[,])input.Clone();
                    plus[i, j] += epsilon;
                    var minus = (float[,])input.Clone();
                    minus[i, j] -= epsilon;

                    float lossPlus = Loss(new RMSNorm(dim, epsilon: 1e-6f).Forward(plus), upstream);
                    float lossMinus = Loss(new RMSNorm(dim, epsilon: 1e-6f).Forward(minus), upstream);

                    float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);

                    Assert.True(
                        MathF.Abs(numericalGrad - analyticalGrad[i, j]) < 1e-2f,
                        $"Mismatch at ({i},{j}): analytical={analyticalGrad[i, j]}, numerical={numericalGrad}");
                }
            }
        }

        [Fact]
        public void Backward_WeightGradient_MatchesNumericalGradient()
        {
            int dim = 4;
            var input = new float[,] { { 0.8f, -0.4f, 1.6f, -1.1f } };
            var upstream = new float[,] { { 0.2f, 0.4f, -0.3f, 0.1f } };
            float epsilon = 1e-3f;

            for (int j = 0; j < dim; j++)
            {
                var weightPlus = new float[dim];
                var weightMinus = new float[dim];
                for (int k = 0; k < dim; k++)
                {
                    weightPlus[k] = 1f;
                    weightMinus[k] = 1f;
                }
                weightPlus[j] += epsilon;
                weightMinus[j] -= epsilon;

                var normPlus = WithWeight(dim, weightPlus);
                float lossPlus = Loss(normPlus.Forward(input), upstream);

                var normMinus = WithWeight(dim, weightMinus);
                float lossMinus = Loss(normMinus.Forward(input), upstream);

                float numericalGrad = (lossPlus - lossMinus) / (2 * epsilon);

                var norm = new RMSNorm(dim, epsilon: 1e-6f);
                norm.Forward(input);
                norm.Backward(upstream);
                norm.AverageGradients(1);
                float analyticalGrad = norm.WeightGradientAt(j);

                Assert.True(
                    MathF.Abs(numericalGrad - analyticalGrad) < 1e-2f,
                    $"Weight grad mismatch at {j}: analytical={analyticalGrad}, numerical={numericalGrad}");
            }
        }

        [Fact]
        public void ForwardBatch_MatchesForwardAppliedPerRow()
        {
            int dim = 4;
            var batch = new float[,,]
            {
                { { 1f, 2f, 3f, 4f }, { -1f, 0.5f, 2f, -3f } },
                { { 0.1f, 0.2f, 0.3f, 0.4f }, { 5f, -5f, 5f, -5f } }
            };

            var normBatch = new RMSNorm(dim, epsilon: 1e-6f);
            var normSingle = new RMSNorm(dim, epsilon: 1e-6f);

            var batchOutput = normBatch.ForwardBatch(batch);

            for (int b = 0; b < 2; b++)
            {
                for (int s = 0; s < 2; s++)
                {
                    var row = new float[1, dim];
                    for (int j = 0; j < dim; j++) row[0, j] = batch[b, s, j];

                    var rowOutput = normSingle.Forward(row);

                    for (int j = 0; j < dim; j++)
                    {
                        Assert.Equal(rowOutput[0, j], batchOutput[b, s, j], precision: 4);
                    }
                }
            }
        }

        [Fact]
        public void BackwardBatch_MatchesBackwardAppliedPerRow()
        {
            int dim = 3;
            var batch = new float[,,]
            {
                { { 1f, -2f, 0.5f } },
                { { 3f, 1f, -1f } }
            };
            var upstream = new float[,,]
            {
                { { 0.1f, 0.2f, -0.3f } },
                { { -0.1f, 0.4f, 0.2f } }
            };

            var normBatch = new RMSNorm(dim, epsilon: 1e-6f);
            normBatch.ForwardBatch(batch);
            var gradBatch = normBatch.BackwardBatch(upstream);

            for (int b = 0; b < 2; b++)
            {
                var row = new float[1, dim];
                var upRow = new float[1, dim];
                for (int j = 0; j < dim; j++)
                {
                    row[0, j] = batch[b, 0, j];
                    upRow[0, j] = upstream[b, 0, j];
                }

                var normSingle = new RMSNorm(dim, epsilon: 1e-6f);
                normSingle.Forward(row);
                var gradRow = normSingle.Backward(upRow);

                for (int j = 0; j < dim; j++)
                {
                    Assert.Equal(gradRow[0, j], gradBatch[b, 0, j], precision: 4);
                }
            }
        }

        private static float Loss(float[,] output, float[,] upstream)
        {
            float total = 0f;
            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int j = 0; j < output.GetLength(1); j++)
                {
                    total += output[i, j] * upstream[i, j];
                }
            }
            return total;
        }

        private static RMSNorm WithWeight(int dim, float[] weight)
        {
            var state = new RMSNorm(dim, epsilon: 1e-6f).SaveState();
            state.Weight = weight;
            return RMSNorm.LoadState(state);
        }
    }
}
