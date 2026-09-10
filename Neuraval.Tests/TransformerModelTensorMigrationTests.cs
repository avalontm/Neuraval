using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class TransformerModelTensorMigrationTests
    {
        private const float Tolerance = 1e-3f;

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
            var model = new TransformerModel(
                vocabSize: 12, embeddingDim: 8, numLayers: 1, numHeads: 2,
                feedforwardDim: 16, maxSequenceLength: 6, dropout: 0.1f, seed: 5);

            var tokens = new[] { 1, 4, 7 };

            var first = model.Forward(tokens, training: false);
            var second = model.Forward(tokens, training: false);

            AssertMatricesEqual(first, second);
        }

        [Fact]
        public void Forward_NoBlocks_MatchesManualEmbeddingNormAndProjection()
        {
            const int vocabSize = 10;
            const int embeddingDim = 6;

            var model = new TransformerModel(
                vocabSize: vocabSize, embeddingDim: embeddingDim, numLayers: 0, numHeads: 2,
                feedforwardDim: 12, maxSequenceLength: 5, dropout: 0.0f, seed: 3);

            var tokens = new[] { 2, 5, 9 };
            var logits = model.Forward(tokens, training: false);

            var state = model.SaveState();
            var embeddings = Unflatten(state.EmbeddingState.Embeddings, vocabSize, embeddingDim);
            var encodings = new PositionalEncoding(5, embeddingDim).GetEncodings(tokens.Length);

            var hidden = new float[tokens.Length, embeddingDim];
            for (int i = 0; i < tokens.Length; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    hidden[i, j] = embeddings[tokens[i], j] + encodings[i, j];
                }
            }

            var normalized = LayerNormReference(hidden, epsilon: 1e-5f);

            var expectedLogits = new float[tokens.Length, vocabSize];
            for (int i = 0; i < tokens.Length; i++)
            {
                for (int v = 0; v < vocabSize; v++)
                {
                    float sum = state.OutputBias[v];
                    for (int k = 0; k < embeddingDim; k++)
                    {
                        sum += normalized[i, k] * embeddings[v, k];
                    }
                    expectedLogits[i, v] = sum;
                }
            }

            AssertMatricesEqual(expectedLogits, logits);
        }

        [Fact]
        public void ForwardBatch_MatchesForwardPerSequence_WithoutPadding()
        {
            const int seqLen = 4;

            var model = new TransformerModel(
                vocabSize: 15, embeddingDim: 8, numLayers: 2, numHeads: 2,
                feedforwardDim: 16, maxSequenceLength: 6, dropout: 0.0f, seed: 9);

            var tokensA = new[] { 1, 2, 3, 4 };
            var tokensB = new[] { 5, 6, 7, 8 };

            var expectedA = model.Forward(tokensA, training: false);
            var expectedB = model.Forward(tokensB, training: false);

            var tokenBatch = new int[2, seqLen];
            for (int i = 0; i < seqLen; i++)
            {
                tokenBatch[0, i] = tokensA[i];
                tokenBatch[1, i] = tokensB[i];
            }

            var validLengths = new[] { seqLen, seqLen };
            var actualBatch = model.ForwardBatch(tokenBatch, validLengths, training: false);

            for (int i = 0; i < seqLen; i++)
            {
                for (int v = 0; v < 15; v++)
                {
                    Assert.True(MathF.Abs(expectedA[i, v] - actualBatch[0, i, v]) < Tolerance);
                    Assert.True(MathF.Abs(expectedB[i, v] - actualBatch[1, i, v]) < Tolerance);
                }
            }
        }

        [Fact]
        public void CalculateCausalLoss_ThenUpdateWeights_ProducesNonZeroGradientsAndChangesLogits()
        {
            var model = new TransformerModel(
                vocabSize: 10, embeddingDim: 8, numLayers: 1, numHeads: 2,
                feedforwardDim: 16, maxSequenceLength: 6, dropout: 0.0f, seed: 17);

            var sequence = new[] { 1, 2, 3, 4, 5 };

            model.ZeroGradients();
            var loss = model.CalculateCausalLoss(sequence, lossStartIndex: 0);
            model.AverageGradients(1);

            Assert.True(loss > 0f);

            var beforeUpdate = model.Forward(sequence, training: false);
            model.UpdateWeights(0.05f);
            var afterUpdate = model.Forward(sequence, training: false);

            bool changed = false;
            for (int i = 0; i < beforeUpdate.GetLength(0) && !changed; i++)
            {
                for (int j = 0; j < beforeUpdate.GetLength(1) && !changed; j++)
                {
                    if (MathF.Abs(beforeUpdate[i, j] - afterUpdate[i, j]) > 1e-6f)
                    {
                        changed = true;
                    }
                }
            }

            Assert.True(changed);
        }

        [Fact]
        public void SaveState_ThenLoadState_ProducesSameLogits()
        {
            var model = new TransformerModel(
                vocabSize: 10, embeddingDim: 8, numLayers: 1, numHeads: 2,
                feedforwardDim: 16, maxSequenceLength: 6, dropout: 0.0f, seed: 23);

            var tokens = new[] { 1, 2, 3 };
            var expected = model.Forward(tokens, training: false);

            var state = model.SaveState();
            var restored = TransformerModel.LoadState(state);
            var actual = restored.Forward(tokens, training: false);

            AssertMatricesEqual(expected, actual);
        }

        private static float[,] Unflatten(float[] flat, int rows, int cols)
        {
            var matrix = new float[rows, cols];
            int index = 0;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = flat[index++];
                }
            }

            return matrix;
        }

        private static float[,] LayerNormReference(float[,] input, float epsilon)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                float mean = 0f;
                for (int j = 0; j < cols; j++)
                {
                    mean += input[i, j];
                }
                mean /= cols;

                float variance = 0f;
                for (int j = 0; j < cols; j++)
                {
                    float diff = input[i, j] - mean;
                    variance += diff * diff;
                }
                variance /= cols;

                float std = MathF.Sqrt(variance + epsilon);

                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = (input[i, j] - mean) / std;
                }
            }

            return result;
        }
    }
}
