using System;
using System.Linq;
using Neuraval.Core.Generation;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class TextGeneratorTests
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
        public void Constructor_NullModel_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new TextGenerator(null!));
        }

        [Fact]
        public void Generate_NullPrompt_Throws()
        {
            var generator = new TextGenerator(new ModernDecoderModel(SmallConfig(), seed: 1));
            var options = GenerationOptions.CreateGreedy(3, Array.Empty<int>());

            Assert.Throws<ArgumentException>(() => generator.Generate(null!, options));
        }

        [Fact]
        public void Generate_EmptyPrompt_Throws()
        {
            var generator = new TextGenerator(new ModernDecoderModel(SmallConfig(), seed: 1));
            var options = GenerationOptions.CreateGreedy(3, Array.Empty<int>());

            Assert.Throws<ArgumentException>(() => generator.Generate(Array.Empty<int>(), options));
        }

        [Fact]
        public void Generate_NullOptions_Throws()
        {
            var generator = new TextGenerator(new ModernDecoderModel(SmallConfig(), seed: 1));

            Assert.Throws<ArgumentNullException>(() => generator.Generate(new[] { 1 }, null!));
        }

        [Fact]
        public void Generate_NoStopTokenSampled_StopsAtMaxNewTokens()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 1);
            var generator = new TextGenerator(model);
            var options = GenerationOptions.CreateGreedy(5, Array.Empty<int>());

            var result = generator.Generate(new[] { 1, 2, 3 }, options);

            Assert.Equal(5, result.GeneratedTokenIds.Count);
            Assert.Equal(GenerationFinishReason.MaxNewTokens, result.FinishReason);
        }

        [Fact]
        public void Generate_StopTokenMatchesFirstGreedyToken_StopsImmediately()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 1);
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

            var result = generator.Generate(prompt, options);

            Assert.Single(result.GeneratedTokenIds);
            Assert.Equal(expectedToken, result.GeneratedTokenIds[0]);
            Assert.Equal(GenerationFinishReason.StopToken, result.FinishReason);
        }

        [Fact]
        public void Generate_AllGeneratedTokensWithinVocabRange()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 2);
            var generator = new TextGenerator(model);
            var options = GenerationOptions.Create(false, 0.9f, 1f, model.VocabSize, 1.1f, 20, Array.Empty<int>(), seed: 3);

            var result = generator.Generate(new[] { 1, 2 }, options);

            Assert.All(result.GeneratedTokenIds, token => Assert.InRange(token, 0, model.VocabSize - 1));
        }

        [Fact]
        public void Generate_GreedyIsDeterministicAcrossRuns()
        {
            var model = new ModernDecoderModel(SmallConfig(), seed: 4);
            var options = GenerationOptions.CreateGreedy(6, Array.Empty<int>());

            var firstRun = new TextGenerator(model).Generate(new[] { 1, 2, 3 }, options, new Random(10));
            var secondRun = new TextGenerator(model).Generate(new[] { 1, 2, 3 }, options, new Random(99));

            Assert.Equal(firstRun.GeneratedTokenIds, secondRun.GeneratedTokenIds);
        }

        [Fact]
        public void Generate_PromptLongerThanContextWindow_DoesNotThrow()
        {
            var model = new ModernDecoderModel(SmallConfig(maxPositionEmbeddings: 4), seed: 1);
            var generator = new TextGenerator(model);
            var options = GenerationOptions.CreateGreedy(8, Array.Empty<int>());

            var result = generator.Generate(new[] { 1, 2, 3, 4 }, options);

            Assert.Equal(8, result.GeneratedTokenIds.Count);
        }
    }
}
