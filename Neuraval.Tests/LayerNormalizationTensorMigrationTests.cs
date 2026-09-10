using System;
using Neuraval.Core.Models;
using Neuraval.Core.Utils;
using Xunit;

namespace Neuraval.Tests
{
    public class LayerNormalizationTensorMigrationTests
    {
        private const float Tolerance = 1e-4f;

        private static float[,] RandomMatrix(int rows, int cols, int seed)
        {
            var random = new Random(seed);
            var matrix = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }

            return matrix;
        }

        private static void AssertMatricesEqual(float[,] expected, float[,] actual)
        {
            Assert.Equal(expected.GetLength(0), actual.GetLength(0));
            Assert.Equal(expected.GetLength(1), actual.GetLength(1));

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    Assert.True(MathF.Abs(expected[i, j] - actual[i, j]) < Tolerance,
                        $"Diferencia en [{i},{j}]: esperado {expected[i, j]}, obtenido {actual[i, j]}");
                }
            }
        }

        [Fact]
        public void Forward_MatchesParallelLayerNormRows()
        {
            var input = RandomMatrix(6, 8, 1);
            var layer = new LayerNormalization(8);
            var state = layer.SaveState();

            var expected = Matematicas.ParallelLayerNormRows(input, state.Gamma, state.Beta, state.Epsilon, out _, out _);
            var actual = layer.Forward(input);

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void ForwardBatch_MatchesForwardPerRow()
        {
            var batch = new float[2, 4, 5];
            var random = new Random(2);

            for (int b = 0; b < 2; b++)
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 5; j++)
                        batch[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);

            var layer = new LayerNormalization(5);

            var singleInput0 = new float[4, 5];
            var singleInput1 = new float[4, 5];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 5; j++)
                {
                    singleInput0[i, j] = batch[0, i, j];
                    singleInput1[i, j] = batch[1, i, j];
                }

            var expected0 = layer.Forward(singleInput0);
            var expected1 = layer.Forward(singleInput1);

            var actualBatch = layer.ForwardBatch(batch);

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    Assert.True(MathF.Abs(expected0[i, j] - actualBatch[0, i, j]) < Tolerance);
                    Assert.True(MathF.Abs(expected1[i, j] - actualBatch[1, i, j]) < Tolerance);
                }
            }
        }

        [Fact]
        public void Forward_ThenBackward_ProducesNonZeroGradients()
        {
            var input = RandomMatrix(4, 6, 3);
            var gradOutput = RandomMatrix(4, 6, 4);
            var layer = new LayerNormalization(6);

            layer.Forward(input);
            var gradInput = layer.Backward(gradOutput, 0.001f);
            layer.AverageGradients(1);

            Assert.Equal(4, gradInput.GetLength(0));
            Assert.Equal(6, gradInput.GetLength(1));
            Assert.True(layer.SumSquaredGradients() > 0f);
        }

        [Fact]
        public void Backward_MatchesManualOracle()
        {
            var input = RandomMatrix(5, 7, 11);
            var gradOutput = RandomMatrix(5, 7, 12);
            var layer = new LayerNormalization(7);

            layer.Forward(input);
            var gradInput = layer.Backward(gradOutput, 0.001f);
            layer.AverageGradients(1);

            var state = layer.SaveState();
            var (expectedGradInput, expectedGammaGrad, expectedBetaGrad) =
                ManualLayerNormBackward(input, gradOutput, state.Gamma, state.Epsilon);

            AssertMatricesEqual(expectedGradInput, gradInput);

            for (int j = 0; j < 7; j++)
            {
                Assert.True(MathF.Abs(expectedGammaGrad[j] - layer.GammaGradientAt(j)) < Tolerance);
                Assert.True(MathF.Abs(expectedBetaGrad[j] - layer.BetaGradientAt(j)) < Tolerance);
            }
        }

        [Fact]
        public void BackwardBatch_MatchesBackwardPerSequence()
        {
            const int batchSize = 2;
            const int seqLen = 4;
            const int dim = 5;

            var batchInput = new float[batchSize, seqLen, dim];
            var batchGradOutput = new float[batchSize, seqLen, dim];
            var random = new Random(21);

            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < dim; j++)
                    {
                        batchInput[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                        batchGradOutput[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                    }

            var referenceLayer = new LayerNormalization(dim);
            var expectedGradInput = new float[batchSize, seqLen, dim];

            for (int b = 0; b < batchSize; b++)
            {
                var singleInput = new float[seqLen, dim];
                var singleGradOutput = new float[seqLen, dim];
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < dim; j++)
                    {
                        singleInput[i, j] = batchInput[b, i, j];
                        singleGradOutput[i, j] = batchGradOutput[b, i, j];
                    }

                referenceLayer.Forward(singleInput);
                var singleGradInput = referenceLayer.Backward(singleGradOutput, 0.001f);

                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < dim; j++)
                        expectedGradInput[b, i, j] = singleGradInput[i, j];
            }
            referenceLayer.AverageGradients(batchSize);

            var batchLayer = new LayerNormalization(dim);
            batchLayer.ForwardBatch(batchInput);
            var actualGradInput = batchLayer.BackwardBatch(batchGradOutput, 0.001f);
            batchLayer.AverageGradients(batchSize);

            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < dim; j++)
                        Assert.True(MathF.Abs(expectedGradInput[b, i, j] - actualGradInput[b, i, j]) < Tolerance);

            for (int j = 0; j < dim; j++)
            {
                Assert.True(MathF.Abs(referenceLayer.GammaGradientAt(j) - batchLayer.GammaGradientAt(j)) < Tolerance);
                Assert.True(MathF.Abs(referenceLayer.BetaGradientAt(j) - batchLayer.BetaGradientAt(j)) < Tolerance);
            }
        }

        private static (float[,] gradInput, float[] gammaGrad, float[] betaGrad) ManualLayerNormBackward(
            float[,] input, float[,] gradOutput, float[] gamma, float epsilon)
        {
            int rows = input.GetLength(0);
            int dim = input.GetLength(1);

            var gradInput = new float[rows, dim];
            var gammaGrad = new float[dim];
            var betaGrad = new float[dim];

            for (int i = 0; i < rows; i++)
            {
                float mean = 0;
                for (int j = 0; j < dim; j++)
                    mean += input[i, j];
                mean /= dim;

                float variance = 0;
                for (int j = 0; j < dim; j++)
                    variance += (input[i, j] - mean) * (input[i, j] - mean);
                variance /= dim;
                float std = MathF.Sqrt(variance + epsilon);

                var normalized = new float[dim];
                for (int j = 0; j < dim; j++)
                {
                    normalized[j] = (input[i, j] - mean) / std;
                    gammaGrad[j] += gradOutput[i, j] * normalized[j];
                    betaGrad[j] += gradOutput[i, j];
                }

                float gradMean = 0;
                float gradVar = 0;
                for (int j = 0; j < dim; j++)
                {
                    float gradNorm = gradOutput[i, j] * gamma[j];
                    gradVar += gradNorm * (input[i, j] - mean) * (-0.5f) * MathF.Pow(std, -3f);
                    gradMean += gradNorm * (-1.0f / std);
                }

                for (int j = 0; j < dim; j++)
                {
                    float gradNorm = gradOutput[i, j] * gamma[j];
                    gradInput[i, j] = (gradNorm / std) +
                                      (gradVar * 2 * (input[i, j] - mean) / dim) +
                                      (gradMean / dim);
                }
            }

            return (gradInput, gammaGrad, betaGrad);
        }
    }
}
