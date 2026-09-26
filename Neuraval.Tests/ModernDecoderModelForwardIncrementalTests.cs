using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class ModernDecoderModelForwardIncrementalTests
    {
        private static TransformerConfig SmallConfig(int maxPositionEmbeddings = 16)
        {
            return new TransformerConfig
            {
                VocabSize = 16,
                HiddenSize = 8,
                NumHiddenLayers = 2,
                NumAttentionHeads = 4,
                NumKeyValueHeads = 2,
                IntermediateSize = 12,
                MaxPositionEmbeddings = maxPositionEmbeddings,
                RopeTheta = 10000f,
                RmsNormEps = 1e-6f,
                TieWordEmbeddings = true
            };
        }

        private static void AssertRowsClose(float[,] expected, int expectedRow, float[,] actual, int actualRow, int vocabSize)
        {
            for (int v = 0; v < vocabSize; v++)
                Assert.True(
                    MathF.Abs(expected[expectedRow, v] - actual[actualRow, v]) < 1e-3f,
                    $"Diferencia en v={v}: esperado {expected[expectedRow, v]}, obtenido {actual[actualRow, v]}");
        }

        [Fact]
        public void ForwardIncremental_TokenByToken_MatchesFullForward()
        {
            var config = SmallConfig();
            var model = new ModernDecoderModel(config, seed: 5);
            var tokens = new[] { 1, 4, 7, 2, 9 };

            var fullLogits = model.Forward(tokens);
            var cache = model.CreateGenerationCache(tokens.Length);

            for (int i = 0; i < tokens.Length; i++)
            {
                var stepLogits = model.ForwardIncremental(new[] { tokens[i] }, cache);
                AssertRowsClose(fullLogits, i, stepLogits, 0, config.VocabSize);
            }
        }

        [Fact]
        public void ForwardIncremental_PrefillThenDecode_MatchesFullForward()
        {
            var config = SmallConfig();
            var model = new ModernDecoderModel(config, seed: 6);
            var tokens = new[] { 2, 3, 5, 8, 1, 6 };
            int prefillLength = 3;

            var fullLogits = model.Forward(tokens);
            var cache = model.CreateGenerationCache(tokens.Length);

            var prefillTokens = new int[prefillLength];
            Array.Copy(tokens, prefillTokens, prefillLength);

            var prefillLogits = model.ForwardIncremental(prefillTokens, cache);

            for (int i = 0; i < prefillLength; i++)
                AssertRowsClose(fullLogits, i, prefillLogits, i, config.VocabSize);

            for (int i = prefillLength; i < tokens.Length; i++)
            {
                var stepLogits = model.ForwardIncremental(new[] { tokens[i] }, cache);
                AssertRowsClose(fullLogits, i, stepLogits, 0, config.VocabSize);
            }
        }

        [Fact]
        public void CreateGenerationCache_CapacityAboveMaxPositionEmbeddings_Throws()
        {
            var model = new ModernDecoderModel(SmallConfig(maxPositionEmbeddings: 4), seed: 1);

            Assert.Throws<ArgumentException>(() => model.CreateGenerationCache(5));
        }

        [Fact]
        public void ForwardIncremental_ExceedingMaxPositionEmbeddings_Throws()
        {
            var model = new ModernDecoderModel(SmallConfig(maxPositionEmbeddings: 4), seed: 1);
            var cache = model.CreateGenerationCache(4);

            model.ForwardIncremental(new[] { 1, 2, 3 }, cache);

            Assert.Throws<ArgumentException>(() => model.ForwardIncremental(new[] { 4, 5 }, cache));
        }

        [Fact]
        public void ForwardIncremental_NullCache_Throws()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 1);

            Assert.Throws<ArgumentNullException>(() => model.ForwardIncremental(new[] { 1 }, null!));
        }

        [Fact]
        public void ForwardIncremental_EmptyTokens_Throws()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 1);
            var cache = model.CreateGenerationCache(4);

            Assert.Throws<ArgumentException>(() => model.ForwardIncremental(Array.Empty<int>(), cache));
        }
    }
}
