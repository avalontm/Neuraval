using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class FeedForwardNetworkOutputDimTests
    {
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

        [Fact]
        public void OmittedOutputDim_FallsBackToEmbeddingDim()
        {
            var network = new FeedForwardNetwork(embeddingDim: 6, hiddenDim: 10, seed: 3);

            Assert.Equal(6, network.OutputDim);
        }

        [Fact]
        public void Forward_ProducesExactlyOutputDimColumns()
        {
            var input = RandomMatrix(4, 6, 11);
            var network = new FeedForwardNetwork(embeddingDim: 6, hiddenDim: 10, seed: 3, outputDim: 2);

            var output = network.Forward(input);

            Assert.Equal(4, output.GetLength(0));
            Assert.Equal(2, output.GetLength(1));
        }

        [Fact]
        public void ForwardBatch_ProducesExactlyOutputDimColumns()
        {
            var network = new FeedForwardNetwork(embeddingDim: 6, hiddenDim: 10, seed: 3, outputDim: 2);
            var input = new float[2, 3, 6];
            var random = new Random(29);

            for (int b = 0; b < 2; b++)
            {
                for (int t = 0; t < 3; t++)
                {
                    for (int i = 0; i < 6; i++)
                    {
                        input[b, t, i] = (float)(random.NextDouble() * 2.0 - 1.0);
                    }
                }
            }

            var output = network.ForwardBatch(input);

            Assert.Equal(2, output.GetLength(0));
            Assert.Equal(3, output.GetLength(1));
            Assert.Equal(2, output.GetLength(2));
        }

        [Fact]
        public void SaveAndLoad_PreservesOutputDim()
        {
            var network = new FeedForwardNetwork(embeddingDim: 6, hiddenDim: 10, seed: 3, outputDim: 2);

            var loaded = FeedForwardNetwork.LoadState(network.SaveState());

            Assert.Equal(2, loaded.OutputDim);
        }

        [Fact]
        public void LoadState_InfersLegacySevenOutputGenome_FromWeights2Length()
        {
            var state = new FeedForwardNetworkState
            {
                EmbeddingDim = 7,
                HiddenDim = 8,
                Weights1 = new float[7 * 8],
                Bias1 = new float[8],
                Weights2 = new float[8 * 7],
                Bias2 = new float[7]
            };

            var network = FeedForwardNetwork.LoadState(state);

            Assert.Equal(7, network.OutputDim);
        }

        [Fact]
        public void LoadState_InfersTwoOutputGenome_FromWeights2Length()
        {
            var state = new FeedForwardNetworkState
            {
                EmbeddingDim = 7,
                HiddenDim = 8,
                Weights1 = new float[7 * 8],
                Bias1 = new float[8],
                Weights2 = new float[8 * 2],
                Bias2 = new float[2]
            };

            var network = FeedForwardNetwork.LoadState(state);

            Assert.Equal(2, network.OutputDim);
        }

        [Fact]
        public void LoadState_WithInconsistentWeights2_FallsBackToEmbeddingDim()
        {
            var state = new FeedForwardNetworkState
            {
                EmbeddingDim = 6,
                HiddenDim = 10,
                Weights1 = new float[6 * 10],
                Bias1 = new float[10],
                Weights2 = new float[8 * 6],
                Bias2 = new float[6]
            };

            var network = FeedForwardNetwork.LoadState(state);

            Assert.Equal(6, network.OutputDim);
        }

        [Fact]
        public void OutputDim_Two_UsesFewerParametersThanEmbeddingDimSeven()
        {
            var seven = new FeedForwardNetwork(embeddingDim: 7, hiddenDim: 8, seed: 3, outputDim: 7);
            var two = new FeedForwardNetwork(embeddingDim: 7, hiddenDim: 8, seed: 3, outputDim: 2);

            int sevenParams = 7 * 8 + 8 + 8 * 7 + 7;
            int twoParams = 7 * 8 + 8 + 8 * 2 + 2;

            Assert.Equal(127, sevenParams);
            Assert.Equal(82, twoParams);
            Assert.True(twoParams < sevenParams);
        }
    }
}