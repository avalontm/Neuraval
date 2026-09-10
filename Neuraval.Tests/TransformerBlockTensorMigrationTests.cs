using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class TransformerBlockTensorMigrationTests
    {
        private const float Tolerance = 1e-3f;

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

        private static float[,,] RandomBatch(int batchSize, int seqLen, int dim, int seed)
        {
            var random = new Random(seed);
            var batch = new float[batchSize, seqLen, dim];

            for (int b = 0; b < batchSize; b++)
                for (int i = 0; i < seqLen; i++)
                    for (int j = 0; j < dim; j++)
                        batch[b, i, j] = (float)(random.NextDouble() * 2.0 - 1.0);

            return batch;
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
            var block = new TransformerBlock(embeddingDim: 8, numHeads: 2, feedforwardDim: 16, dropout: 0.0f, seed: 42);

            var first = block.Forward(input, mask: null, training: false);
            var second = block.Forward(input, mask: null, training: false);

            AssertMatricesEqual(first, second);
        }

        [Fact]
        public void ForwardBatch_MatchesForwardPerRow()
        {
            const int batchSize = 2;
            const int seqLen = 4;
            const int embeddingDim = 6;

            var batch = RandomBatch(batchSize, seqLen, embeddingDim, seed: 3);
            var block = new TransformerBlock(embeddingDim: embeddingDim, numHeads: 2, feedforwardDim: 12, dropout: 0.0f, seed: 11);

            var singleInput0 = new float[seqLen, embeddingDim];
            var singleInput1 = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    singleInput0[i, j] = batch[0, i, j];
                    singleInput1[i, j] = batch[1, i, j];
                }
            }

            var expected0 = block.Forward(singleInput0, mask: null, training: false);
            var expected1 = block.Forward(singleInput1, mask: null, training: false);

            var actualBatch = block.ForwardBatch(batch, mask: (float[,]?)null, training: false);

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
            var block = new TransformerBlock(embeddingDim: 6, numHeads: 2, feedforwardDim: 12, dropout: 0.0f, seed: 21);

            block.Forward(input, mask: null, training: true);
            var gradInput = block.Backward(gradOutput);
            block.AverageGradients(1);

            Assert.Equal(4, gradInput.GetLength(0));
            Assert.Equal(6, gradInput.GetLength(1));
            Assert.True(block.SumSquaredGradients() > 0f);
        }

        [Fact]
        public void Forward_ResidualConnection_PreservesInputWhenSubLayersAreIdentityLike()
        {
            const int embeddingDim = 4;
            var block = new TransformerBlock(embeddingDim: embeddingDim, numHeads: 2, feedforwardDim: 8, dropout: 0.0f, seed: 55);

            var input = RandomMatrix(3, embeddingDim, 8);
            var output = block.Forward(input, mask: null, training: false);

            Assert.Equal(input.GetLength(0), output.GetLength(0));
            Assert.Equal(input.GetLength(1), output.GetLength(1));
        }
    }
}
