using System;
using Neuraval.Core.Models;
using Neuraval.Core.Models.RoPE;
using Xunit;

namespace Neuraval.Tests
{
    public class GQAAttentionForwardIncrementalTests
    {
        private static float[,,] RandomInput(int batch, int seq, int dim, int seed)
        {
            var rnd = new Random(seed);
            var input = new float[batch, seq, dim];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seq; s++)
                    for (int d = 0; d < dim; d++)
                        input[b, s, d] = (float)(rnd.NextDouble() * 2 - 1);
            return input;
        }

        private static float[,,] SliceRows(float[,,] source, int startRow, int count)
        {
            int batch = source.GetLength(0);
            int dim = source.GetLength(2);
            var slice = new float[batch, count, dim];
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < count; s++)
                    for (int d = 0; d < dim; d++)
                        slice[b, s, d] = source[b, startRow + s, d];
            return slice;
        }

        private static void AssertRowsClose(float[,,] expected, int expectedRow, float[,,] actual, int actualRow, int dim)
        {
            for (int d = 0; d < dim; d++)
                Assert.True(
                    MathF.Abs(expected[0, expectedRow, d] - actual[0, actualRow, d]) < 1e-4f,
                    $"Diferencia en d={d}: esperado {expected[0, expectedRow, d]}, obtenido {actual[0, actualRow, d]}");
        }

        [Fact]
        public void ForwardIncremental_TokenByToken_MatchesFullForward()
        {
            int hiddenSize = 8;
            int numAttentionHeads = 4;
            int numKeyValueHeads = 2;
            int headDim = hiddenSize / numAttentionHeads;
            int seqLen = 5;

            var rotary = new RotaryEmbedding(new RotaryConfig
            {
                HeadDim = headDim,
                MaxPositionEmbeddings = 16,
                RopeTheta = 10000f
            });

            var attention = new GQAAttention(hiddenSize, numAttentionHeads, numKeyValueHeads, rotary, seed: 3);
            var input = RandomInput(1, seqLen, hiddenSize, seed: 11);

            var fullOutput = attention.Forward(input);

            var incrementalAttention = GQAAttention.LoadState(attention.SaveState(), rotary);
            var cache = new GqaKeyValueCacheLayer(batchSize: 1, capacity: seqLen, numKeyValueHeads: numKeyValueHeads, headDim: headDim);

            for (int position = 0; position < seqLen; position++)
            {
                var stepInput = SliceRows(input, position, 1);
                var stepOutput = incrementalAttention.ForwardIncremental(stepInput, position, cache);

                AssertRowsClose(fullOutput, position, stepOutput, 0, hiddenSize);
            }
        }

        [Fact]
        public void ForwardIncremental_PrefillThenDecode_MatchesFullForward()
        {
            int hiddenSize = 8;
            int numAttentionHeads = 4;
            int numKeyValueHeads = 2;
            int headDim = hiddenSize / numAttentionHeads;
            int seqLen = 6;
            int prefillLength = 4;

            var rotary = new RotaryEmbedding(new RotaryConfig
            {
                HeadDim = headDim,
                MaxPositionEmbeddings = 16,
                RopeTheta = 10000f
            });

            var attention = new GQAAttention(hiddenSize, numAttentionHeads, numKeyValueHeads, rotary, seed: 4);
            var input = RandomInput(1, seqLen, hiddenSize, seed: 21);

            var fullOutput = attention.Forward(input);

            var incrementalAttention = GQAAttention.LoadState(attention.SaveState(), rotary);
            var cache = new GqaKeyValueCacheLayer(batchSize: 1, capacity: seqLen, numKeyValueHeads: numKeyValueHeads, headDim: headDim);

            var prefillInput = SliceRows(input, 0, prefillLength);
            var prefillOutput = incrementalAttention.ForwardIncremental(prefillInput, 0, cache);

            for (int position = 0; position < prefillLength; position++)
                AssertRowsClose(fullOutput, position, prefillOutput, position, hiddenSize);

            for (int position = prefillLength; position < seqLen; position++)
            {
                var stepInput = SliceRows(input, position, 1);
                var stepOutput = incrementalAttention.ForwardIncremental(stepInput, position, cache);

                AssertRowsClose(fullOutput, position, stepOutput, 0, hiddenSize);
            }
        }

        [Fact]
        public void ForwardIncremental_AppendsExactlyOneEntryPerCall()
        {
            int hiddenSize = 4;
            int numAttentionHeads = 2;
            int numKeyValueHeads = 1;
            int headDim = hiddenSize / numAttentionHeads;

            var attention = new GQAAttention(hiddenSize, numAttentionHeads, numKeyValueHeads, seed: 9);
            var cache = new GqaKeyValueCacheLayer(batchSize: 1, capacity: 3, numKeyValueHeads: numKeyValueHeads, headDim: headDim);
            var input = RandomInput(1, 1, hiddenSize, seed: 30);

            attention.ForwardIncremental(input, 0, cache);
            Assert.Equal(1, cache.Length);

            attention.ForwardIncremental(input, 1, cache);
            Assert.Equal(2, cache.Length);
        }

        [Fact]
        public void ForwardIncremental_IncompatibleCacheShape_Throws()
        {
            var attention = new GQAAttention(hiddenSize: 8, numAttentionHeads: 4, numKeyValueHeads: 2, seed: 1);
            var wrongCache = new GqaKeyValueCacheLayer(batchSize: 1, capacity: 4, numKeyValueHeads: 1, headDim: 2);
            var input = RandomInput(1, 1, 8, seed: 1);

            Assert.Throws<ArgumentException>(() => attention.ForwardIncremental(input, 0, wrongCache));
        }

        [Fact]
        public void ForwardIncremental_NullCache_Throws()
        {
            var attention = new GQAAttention(hiddenSize: 8, numAttentionHeads: 4, numKeyValueHeads: 2, seed: 1);
            var input = RandomInput(1, 1, 8, seed: 1);

            Assert.Throws<ArgumentNullException>(() => attention.ForwardIncremental(input, 0, null!));
        }
    }
}
