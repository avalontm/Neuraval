using System;
using Neuraval.Core.Models;
using Neuraval.Core.Utils;
using Xunit;

namespace Neuraval.Tests
{
    public class BatchVsIndividualForwardTests
    {
        private const int VocabSize = 15;
        private const int EmbeddingDim = 8;
        private const int NumLayers = 2;
        private const int NumHeads = 2;
        private const int FeedforwardDim = 16;
        private const int MaxSequenceLength = 10;
        private const float Tolerance = 1e-4f;

        private static TransformerModel CreateModel(int seed = 77)
        {
            return new TransformerModel(
                vocabSize: VocabSize,
                embeddingDim: EmbeddingDim,
                numLayers: NumLayers,
                numHeads: NumHeads,
                feedforwardDim: FeedforwardDim,
                maxSequenceLength: MaxSequenceLength,
                dropout: 0.0f,
                seed: seed);
        }

        [Fact]
        public void ForwardBatch_MatchesIndividualForward_NoPadding()
        {
            Matematicas.SetNumThreads(1);
            var model = CreateModel();

            int batchSize = 4;
            int seqLen = 6;

            var random = new Random(42);
            var tokenBatch = new int[batchSize, seqLen];
            var sequences = new int[batchSize][];
            var validLengths = new int[batchSize];

            for (int b = 0; b < batchSize; b++)
            {
                sequences[b] = new int[seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    int token = random.Next(0, VocabSize);
                    tokenBatch[b, i] = token;
                    sequences[b][i] = token;
                }
                validLengths[b] = seqLen;
            }

            var batchLogits = model.ForwardBatch(tokenBatch, validLengths, training: false);

            for (int b = 0; b < batchSize; b++)
            {
                var individualLogits = model.Forward(sequences[b], training: false);

                for (int i = 0; i < seqLen; i++)
                {
                    for (int v = 0; v < VocabSize; v++)
                    {
                        float expected = individualLogits[i, v];
                        float actual = batchLogits[b, i, v];
                        Assert.True(
                            MathF.Abs(expected - actual) < Tolerance,
                            $"batch[{b},{i},{v}]={actual} difiere de individual[{i},{v}]={expected}");
                    }
                }
            }
        }
    }
}
