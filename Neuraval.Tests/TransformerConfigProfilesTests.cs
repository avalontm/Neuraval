using Neuraval.Core.Models;
using Neuraval.Core.Models.Architectures;
using Xunit;

namespace Neuraval.Tests
{
    public class TransformerConfigProfilesTests
    {
        [Fact]
        public void Nano_MatchesRoadmapValues()
        {
            var config = TransformerConfigProfiles.Nano(vocabSize: 32000);

            Assert.Equal(256, config.HiddenSize);
            Assert.Equal(6, config.NumHiddenLayers);
            Assert.Equal(8, config.NumAttentionHeads);
            Assert.Equal(2, config.NumKeyValueHeads);
            Assert.Equal(768, config.IntermediateSize);
            Assert.Equal(2048, config.MaxPositionEmbeddings);
        }

        [Fact]
        public void Small_MatchesRoadmapValues()
        {
            var config = TransformerConfigProfiles.Small(vocabSize: 32000);

            Assert.Equal(512, config.HiddenSize);
            Assert.Equal(12, config.NumHiddenLayers);
            Assert.Equal(8, config.NumAttentionHeads);
            Assert.Equal(2, config.NumKeyValueHeads);
            Assert.Equal(1408, config.IntermediateSize);
            Assert.Equal(4096, config.MaxPositionEmbeddings);
        }

        [Fact]
        public void Create_DispatchesToMatchingProfile()
        {
            var nano = TransformerConfigProfiles.Create(ModelProfile.Nano, vocabSize: 1000);
            var small = TransformerConfigProfiles.Create(ModelProfile.Small, vocabSize: 1000);

            Assert.Equal(256, nano.HiddenSize);
            Assert.Equal(512, small.HiddenSize);
        }

        [Fact]
        public void Nano_VocabSizeIsConfigurable()
        {
            var config = TransformerConfigProfiles.Nano(vocabSize: 500, seed: 7);

            Assert.Equal(500, config.VocabSize);
            Assert.Equal(7, config.Seed);
        }

        [Fact]
        public void Nano_BuildsModernDecoderModel_ForwardShapeMatchesVocab()
        {
            var config = TransformerConfigProfiles.Nano(vocabSize: 64);
            var model = new ModernDecoderModel(config, seed: config.Seed);
            var tokens = new[] { 1, 4, 7, 2 };

            var logits = model.Forward(tokens);

            Assert.Equal(tokens.Length, logits.GetLength(0));
            Assert.Equal(config.VocabSize, logits.GetLength(1));
        }

        [Fact]
        public void Small_BuildsModernDecoderModel_ForwardShapeMatchesVocab()
        {
            var config = TransformerConfigProfiles.Small(vocabSize: 64);
            var model = new ModernDecoderModel(config, seed: config.Seed);
            var tokens = new[] { 1, 4, 7, 2 };

            var logits = model.Forward(tokens);

            Assert.Equal(tokens.Length, logits.GetLength(0));
            Assert.Equal(config.VocabSize, logits.GetLength(1));
        }

        [Fact]
        public void Nano_ForwardDoesNotProduceNaN()
        {
            var config = TransformerConfigProfiles.Nano(vocabSize: 64);
            var model = new ModernDecoderModel(config, seed: config.Seed);
            var tokens = new[] { 0, 5, 9, 3, 11 };

            var logits = model.Forward(tokens);

            for (int i = 0; i < logits.GetLength(0); i++)
                for (int j = 0; j < logits.GetLength(1); j++)
                    Assert.False(float.IsNaN(logits[i, j]));
        }

        [Fact]
        public void Small_ForwardDoesNotProduceNaN()
        {
            var config = TransformerConfigProfiles.Small(vocabSize: 64);
            var model = new ModernDecoderModel(config, seed: config.Seed);
            var tokens = new[] { 0, 5, 9, 3, 11 };

            var logits = model.Forward(tokens);

            for (int i = 0; i < logits.GetLength(0); i++)
                for (int j = 0; j < logits.GetLength(1); j++)
                    Assert.False(float.IsNaN(logits[i, j]));
        }
    }
}
