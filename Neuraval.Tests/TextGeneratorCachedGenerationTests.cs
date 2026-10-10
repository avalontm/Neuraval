using System;
using Neuraval.Core.Generation;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class TextGeneratorCachedGenerationTests
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

        [Fact]
        public void ForwardIncrementalLastToken_MatchesFinalPromptLogits()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 6);
            var prompt = new[] { 1, 2, 3 };
            var fullCache = model.CreateGenerationCache(prompt.Length);
            var fullLogits = model.ForwardIncremental(prompt, fullCache);

            var lastTokenCache = model.CreateGenerationCache(prompt.Length);
            var lastTokenLogits = model.ForwardIncrementalLastToken(prompt, lastTokenCache);

            for (int token = 0; token < model.VocabSize; token++)
                Assert.Equal(fullLogits[prompt.Length - 1, token], lastTokenLogits[token], precision: 5);
            Assert.Equal(fullCache.Length, lastTokenCache.Length);
        }

        [Fact]
        public void GenerateWithCache_Greedy_MatchesUncachedGenerate()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 7);
            var generator = new TextGenerator(model);
            var options = GenerationOptions.CreateGreedy(6, Array.Empty<int>());

            var uncached = generator.Generate(new[] { 1, 2, 3 }, options);
            var cached = generator.GenerateWithCache(new[] { 1, 2, 3 }, options);

            Assert.Equal(uncached.GeneratedTokenIds, cached.Result.GeneratedTokenIds);
            Assert.Equal(uncached.FinishReason, cached.Result.FinishReason);
        }

        [Fact]
        public void GenerateWithCache_StopToken_MatchesUncachedGenerate()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 8);
            var prompt = new[] { 1, 2, 3 };

            var logits = model.Forward(prompt);
            int lastPosition = logits.GetLength(0) - 1;
            int expectedToken = 0;
            float bestValue = logits[lastPosition, 0];
            for (int i = 1; i < model.VocabSize; i++)
            {
                if (logits[lastPosition, i] > bestValue)
                {
                    bestValue = logits[lastPosition, i];
                    expectedToken = i;
                }
            }

            var generator = new TextGenerator(model);
            var options = GenerationOptions.CreateGreedy(10, new[] { expectedToken });

            var cached = generator.GenerateWithCache(prompt, options);

            Assert.Single(cached.Result.GeneratedTokenIds);
            Assert.Equal(expectedToken, cached.Result.GeneratedTokenIds[0]);
            Assert.Equal(GenerationFinishReason.StopToken, cached.Result.FinishReason);
        }

        [Fact]
        public void GenerateWithCache_Performance_ReportsConsistentTokenCounts()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 9);
            var generator = new TextGenerator(model);
            var options = GenerationOptions.CreateGreedy(4, Array.Empty<int>());

            var cached = generator.GenerateWithCache(new[] { 1, 2, 3 }, options);

            Assert.Equal(3, cached.Performance.PrefillTokenCount);
            Assert.Equal(cached.Result.GeneratedTokenIds.Count, cached.Performance.DecodeTokenCount);
            Assert.True(cached.Performance.CacheMemoryBytes > 0);
        }

        [Fact]
        public void GenerateWithCache_PromptLongerThanContextWindow_DoesNotThrow()
        {
            var model = new ModernDecoderModel(SmallConfig(maxPositionEmbeddings: 4), seed: 1);
            var generator = new TextGenerator(model);
            var options = GenerationOptions.CreateGreedy(8, Array.Empty<int>());

            var cached = generator.GenerateWithCache(new[] { 1, 2, 3, 4 }, options);

            Assert.True(cached.Result.GeneratedTokenIds.Count >= 0);
        }

        [Fact]
        public void GenerateWithCache_NullPrompt_Throws()
        {
            var generator = new TextGenerator(new ModernDecoderModel(SmallConfig(), seed: 1));
            var options = GenerationOptions.CreateGreedy(3, Array.Empty<int>());

            Assert.Throws<ArgumentException>(() => generator.GenerateWithCache(null!, options));
        }

        [Fact]
        public void GenerateWithCache_NullOptions_Throws()
        {
            var generator = new TextGenerator(new ModernDecoderModel(SmallConfig(), seed: 1));

            Assert.Throws<ArgumentNullException>(() => generator.GenerateWithCache(new[] { 1 }, null!));
        }

        [Fact]
        public void GenerateWithCache_SamplingIsDeterministicGivenSameRandom()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 10);
            var options = GenerationOptions.Create(false, 0.9f, 1f, model.VocabSize, 1.1f, 6, Array.Empty<int>(), seed: 3);

            var first = new TextGenerator(model).GenerateWithCache(new[] { 1, 2, 3 }, options, new Random(10));
            var second = new TextGenerator(model).GenerateWithCache(new[] { 1, 2, 3 }, options, new Random(10));

            Assert.Equal(first.Result.GeneratedTokenIds, second.Result.GeneratedTokenIds);
        }
    }
}
