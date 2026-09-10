using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class FeedForwardNetworkTensorMigrationTests
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
        public void Forward_IsDeterministic_AcrossRepeatedCalls()
        {
            var input = RandomMatrix(5, 8, 1);
            var network = new FeedForwardNetwork(embeddingDim: 8, hiddenDim: 16, seed: 42);

            var first = network.Forward(input);
            var second = network.Forward(input);

            AssertMatricesEqual(first, second);
        }

        [Fact]
        public void Forward_MatchesReferenceComputation()
        {
            var input = RandomMatrix(4, 6, 2);
            var network = new FeedForwardNetwork(embeddingDim: 6, hiddenDim: 10, seed: 7);
            var state = network.SaveState();

            var expected = ReferenceForward(input, state.Weights1, state.Bias1, state.Weights2, state.Bias2,
                embeddingDim: 6, hiddenDim: 10);
            var actual = network.Forward(input);

            AssertMatricesEqual(expected, actual);
        }

        [Fact]
        public void ForwardBatch_MatchesForwardPerRow()
        {
            const int batchSize = 2;
            const int seqLen = 4;
            const int embeddingDim = 5;

            var batch = new float[batchSize, seqLen, embeddingDim];
            var random = new Random(3);

            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                        batch[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);

            var network = new FeedForwardNetwork(embeddingDim: embeddingDim, hiddenDim: 9, seed: 11);

            var singleInput0 = new float[seqLen, embeddingDim];
            var singleInput1 = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embeddingDim; j++)
                {
                    singleInput0[i, j] = batch[0, i, j];
                    singleInput1[i, j] = batch[1, i, j];
                }

            var expected0 = network.Forward(singleInput0);
            var expected1 = network.Forward(singleInput1);

            var actualBatch = network.ForwardBatch(batch);

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    Assert.True(MathF.Abs(expected0[i, j] - actualBatch[0, i, j]) < Tolerance);
                    Assert.True(MathF.Abs(expected1[i, j] - actualBatch[1, i, j]) < Tolerance);
                }
            }
        }

        [Fact]
        public void Forward_ThenBackward_ProducesNonZeroGradients()
        {
            var input = RandomMatrix(4, 6, 4);
            var gradOutput = RandomMatrix(4, 6, 5);
            var network = new FeedForwardNetwork(embeddingDim: 6, hiddenDim: 12, seed: 21);

            network.Forward(input);
            var gradInput = network.Backward(gradOutput, 0.001f);
            network.AverageGradients(1);

            Assert.Equal(4, gradInput.GetLength(0));
            Assert.Equal(6, gradInput.GetLength(1));
            Assert.True(network.SumSquaredGradients() > 0f);
        }

        [Fact]
        public void Backward_MatchesManualOracle()
        {
            const int embeddingDim = 6;
            const int hiddenDim = 9;

            var input = RandomMatrix(5, embeddingDim, 31);
            var gradOutput = RandomMatrix(5, embeddingDim, 32);
            var network = new FeedForwardNetwork(embeddingDim: embeddingDim, hiddenDim: hiddenDim, seed: 13);

            network.Forward(input);
            var gradInput = network.Backward(gradOutput, 0.001f);
            network.AverageGradients(1);

            var state = network.SaveState();
            var weights1 = Unflatten(state.Weights1, embeddingDim, hiddenDim);
            var weights2 = Unflatten(state.Weights2, hiddenDim, embeddingDim);

            var (expectedGradInput, expectedGrad1, expectedBiasGrad1, expectedGrad2, expectedBiasGrad2) =
                ManualFeedForwardBackward(input, gradOutput, weights1, state.Bias1, weights2, embeddingDim, hiddenDim);

            AssertMatricesEqual(expectedGradInput, gradInput);

            for (int i = 0; i < embeddingDim; i++)
                for (int j = 0; j < hiddenDim; j++)
                    Assert.True(MathF.Abs(expectedGrad1[i, j] - network.Gradient1At(i, j)) < Tolerance);

            for (int j = 0; j < hiddenDim; j++)
                Assert.True(MathF.Abs(expectedBiasGrad1[j] - network.BiasGradient1At(j)) < Tolerance);

            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < embeddingDim; j++)
                    Assert.True(MathF.Abs(expectedGrad2[i, j] - network.Gradient2At(i, j)) < Tolerance);

            for (int j = 0; j < embeddingDim; j++)
                Assert.True(MathF.Abs(expectedBiasGrad2[j] - network.BiasGradient2At(j)) < Tolerance);
        }

        [Fact]
        public void BackwardBatch_MatchesBackwardPerSequence()
        {
            const int batchSize = 2;
            const int seqLen = 4;
            const int embeddingDim = 5;
            const int hiddenDim = 7;

            var batchInput = new float[batchSize, seqLen, embeddingDim];
            var batchGradOutput = new float[batchSize, seqLen, embeddingDim];
            var random = new Random(41);

            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        batchInput[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                        batchGradOutput[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                    }

            var referenceNetwork = new FeedForwardNetwork(embeddingDim: embeddingDim, hiddenDim: hiddenDim, seed: 55);
            var expectedGradInput = new float[batchSize, seqLen, embeddingDim];

            for (int b = 0; b < batchSize; b++)
            {
                var singleInput = new float[seqLen, embeddingDim];
                var singleGradOutput = new float[seqLen, embeddingDim];
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                    {
                        singleInput[i, j] = batchInput[b, i, j];
                        singleGradOutput[i, j] = batchGradOutput[b, i, j];
                    }

                referenceNetwork.Forward(singleInput);
                var singleGradInput = referenceNetwork.Backward(singleGradOutput, 0.001f);

                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                        expectedGradInput[b, i, j] = singleGradInput[i, j];
            }
            referenceNetwork.AverageGradients(batchSize);

            var batchNetwork = new FeedForwardNetwork(embeddingDim: embeddingDim, hiddenDim: hiddenDim, seed: 55);
            batchNetwork.ForwardBatch(batchInput);
            var actualGradInput = batchNetwork.BackwardBatch(batchGradOutput, 0.001f);
            batchNetwork.AverageGradients(batchSize);

            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < embeddingDim; j++)
                        Assert.True(MathF.Abs(expectedGradInput[b, i, j] - actualGradInput[b, i, j]) < Tolerance);

            for (int i = 0; i < embeddingDim; i++)
                for (int j = 0; j < hiddenDim; j++)
                    Assert.True(MathF.Abs(referenceNetwork.Gradient1At(i, j) - batchNetwork.Gradient1At(i, j)) < Tolerance);

            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < embeddingDim; j++)
                    Assert.True(MathF.Abs(referenceNetwork.Gradient2At(i, j) - batchNetwork.Gradient2At(i, j)) < Tolerance);
        }

        private static float[,] Unflatten(float[] flat, int rows, int cols)
        {
            var matrix = new float[rows, cols];
            int index = 0;

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = flat[index++];

            return matrix;
        }

        private static (float[,] gradInput, float[,] grad1, float[] biasGrad1, float[,] grad2, float[] biasGrad2)
            ManualFeedForwardBackward(float[,] input, float[,] gradOutput, float[,] weights1, float[] bias1,
                float[,] weights2, int embeddingDim, int hiddenDim)
        {
            int seqLen = input.GetLength(0);

            var hidden = new float[seqLen, hiddenDim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < hiddenDim; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < embeddingDim; p++)
                        sum += input[i, p] * weights1[p, j];
                    hidden[i, j] = MathF.Max(0f, sum + bias1[j]);
                }

            var gradHidden = new float[seqLen, hiddenDim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < hiddenDim; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < embeddingDim; p++)
                        sum += gradOutput[i, p] * weights2[j, p];
                    gradHidden[i, j] = hidden[i, j] > 0f ? sum : 0f;
                }

            var grad2 = new float[hiddenDim, embeddingDim];
            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < embeddingDim; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < seqLen; p++)
                        sum += hidden[p, i] * gradOutput[p, j];
                    grad2[i, j] = sum;
                }

            var biasGrad2 = new float[embeddingDim];
            for (int j = 0; j < embeddingDim; j++)
                for (int i = 0; i < seqLen; i++)
                    biasGrad2[j] += gradOutput[i, j];

            var gradInput = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
                for (int j = 0; j < embeddingDim; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < hiddenDim; p++)
                        sum += gradHidden[i, p] * weights1[j, p];
                    gradInput[i, j] = sum;
                }

            var grad1 = new float[embeddingDim, hiddenDim];
            for (int i = 0; i < embeddingDim; i++)
                for (int j = 0; j < hiddenDim; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < seqLen; p++)
                        sum += input[p, i] * gradHidden[p, j];
                    grad1[i, j] = sum;
                }

            var biasGrad1 = new float[hiddenDim];
            for (int j = 0; j < hiddenDim; j++)
                for (int i = 0; i < seqLen; i++)
                    biasGrad1[j] += gradHidden[i, j];

            return (gradInput, grad1, biasGrad1, grad2, biasGrad2);
        }

        private static float[,] ReferenceForward(float[,] input, float[] flatWeights1, float[] bias1,
            float[] flatWeights2, float[] bias2, int embeddingDim, int hiddenDim)
        {
            int seqLen = input.GetLength(0);

            var hidden = new float[seqLen, hiddenDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < hiddenDim; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < embeddingDim; p++)
                    {
                        sum += input[i, p] * flatWeights1[p * hiddenDim + j];
                    }
                    hidden[i, j] = MathF.Max(0f, sum + bias1[j]);
                }
            }

            var output = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    float sum = 0f;
                    for (int p = 0; p < hiddenDim; p++)
                    {
                        sum += hidden[i, p] * flatWeights2[p * embeddingDim + j];
                    }
                    output[i, j] = sum + bias2[j];
                }
            }

            return output;
        }
    }
}
