using System;
using Neuraval.Core.Models;
using Xunit;

namespace Neuraval.Tests
{
    public class ModernDecoderModelTests
    {
        private static TransformerConfig SmallConfig(bool tieWordEmbeddings)
        {
            return new TransformerConfig
            {
                VocabSize = 16,
                HiddenSize = 8,
                NumHiddenLayers = 2,
                NumAttentionHeads = 4,
                NumKeyValueHeads = 2,
                IntermediateSize = 12,
                MaxPositionEmbeddings = 16,
                RopeTheta = 10000f,
                RmsNormEps = 1e-6f,
                TieWordEmbeddings = tieWordEmbeddings
            };
        }

        [Fact]
        public void Forward_OutputShape_MatchesSequenceLengthAndVocabSize()
        {
            var config = SmallConfig(tieWordEmbeddings: true);
            var model = new ModernDecoderModel(config, seed: 1);
            var tokens = new[] { 1, 4, 7, 2 };

            var logits = model.Forward(tokens);

            Assert.Equal(tokens.Length, logits.GetLength(0));
            Assert.Equal(config.VocabSize, logits.GetLength(1));
        }

        [Fact]
        public void Forward_DoesNotProduceNaN()
        {
            var config = SmallConfig(tieWordEmbeddings: false);
            var model = new ModernDecoderModel(config, seed: 2);
            var tokens = new[] { 0, 5, 9, 3, 11 };

            var logits = model.Forward(tokens);

            for (int i = 0; i < logits.GetLength(0); i++)
                for (int j = 0; j < logits.GetLength(1); j++)
                    Assert.False(float.IsNaN(logits[i, j]));
        }

        [Fact]
        public void CalculateCausalLoss_TiedEmbeddings_DecreasesAfterTraining()
        {
            var config = SmallConfig(tieWordEmbeddings: true);
            var model = new ModernDecoderModel(config, seed: 3);
            var tokens = new[] { 2, 5, 8, 3, 1 };

            float firstLoss = model.CalculateCausalLoss(tokens, lossStartIndex: 0);
            model.UpdateWeights(0.1f);

            float lastLoss = firstLoss;
            for (int step = 0; step < 30; step++)
            {
                lastLoss = model.CalculateCausalLoss(tokens, lossStartIndex: 0);
                model.UpdateWeights(0.1f);
            }

            Assert.True(lastLoss < firstLoss, $"La pérdida debe bajar entrenando sobre la misma secuencia: first={firstLoss}, last={lastLoss}");
        }

        [Fact]
        public void CalculateCausalLoss_UntiedEmbeddings_DecreasesAfterTraining()
        {
            var config = SmallConfig(tieWordEmbeddings: false);
            var model = new ModernDecoderModel(config, seed: 4);
            var tokens = new[] { 1, 6, 10, 4, 2 };

            float firstLoss = model.CalculateCausalLoss(tokens, lossStartIndex: 0);
            model.UpdateWeights(0.1f);

            float lastLoss = firstLoss;
            for (int step = 0; step < 30; step++)
            {
                lastLoss = model.CalculateCausalLoss(tokens, lossStartIndex: 0);
                model.UpdateWeights(0.1f);
            }

            Assert.True(lastLoss < firstLoss, $"La pérdida debe bajar entrenando sobre la misma secuencia: first={firstLoss}, last={lastLoss}");
        }

        [Fact]
        public void SaveState_LoadState_ProducesIdenticalForwardOutput()
        {
            var config = SmallConfig(tieWordEmbeddings: false);
            var model = new ModernDecoderModel(config, seed: 5);
            var tokens = new[] { 3, 7, 1, 9 };

            var expected = model.Forward(tokens);
            var reloaded = ModernDecoderModel.LoadState(model.SaveState());
            var actual = reloaded.Forward(tokens);

            for (int i = 0; i < tokens.Length; i++)
                for (int j = 0; j < config.VocabSize; j++)
                    Assert.Equal(expected[i, j], actual[i, j], precision: 4);
        }

        [Fact]
        public void SaveState_LoadState_TiedEmbeddings_ProducesIdenticalForwardOutput()
        {
            var config = SmallConfig(tieWordEmbeddings: true);
            var model = new ModernDecoderModel(config, seed: 6);
            var tokens = new[] { 0, 2, 4, 6 };

            var expected = model.Forward(tokens);
            var reloaded = ModernDecoderModel.LoadState(model.SaveState());
            var actual = reloaded.Forward(tokens);

            for (int i = 0; i < tokens.Length; i++)
                for (int j = 0; j < config.VocabSize; j++)
                    Assert.Equal(expected[i, j], actual[i, j], precision: 4);
        }
    }
}
