using System;
using Neuraval.Core.Models;
using Neuraval.Core.Models.Architectures;
using Xunit;

namespace Neuraval.Tests
{
    public class TransformerConfigTests
    {
        private static TransformerConfig ValidConfig()
        {
            return new TransformerConfig
            {
                Architecture = "legacy",
                VocabSize = 128,
                HiddenSize = 32,
                NumHiddenLayers = 2,
                NumAttentionHeads = 4,
                NumKeyValueHeads = 2,
                IntermediateSize = 64,
                MaxPositionEmbeddings = 16,
                Dropout = 0.0f
            };
        }

        [Fact]
        public void Validate_WithConsistentDimensions_DoesNotThrow()
        {
            var config = ValidConfig();

            var exception = Record.Exception(() => config.Validate());

            Assert.Null(exception);
        }

        [Fact]
        public void Validate_WhenHiddenSizeNotDivisibleByHeads_Throws()
        {
            var config = ValidConfig();
            config.HiddenSize = 30;

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        [Fact]
        public void Validate_WhenAttentionHeadsNotDivisibleByKeyValueHeads_Throws()
        {
            var config = ValidConfig();
            config.NumAttentionHeads = 5;
            config.NumKeyValueHeads = 2;

            Assert.Throws<InvalidOperationException>(() => config.Validate());
        }

        [Fact]
        public void HeadDim_ComputesCorrectly()
        {
            var config = ValidConfig();

            Assert.Equal(8, config.HeadDim);
        }

        [Fact]
        public void NumKeyValueGroups_ComputesCorrectly()
        {
            var config = ValidConfig();

            Assert.Equal(2, config.NumKeyValueGroups);
        }

        [Fact]
        public void Clone_ProducesIndependentCopy()
        {
            var config = ValidConfig();
            var clone = config.Clone();
            clone.HiddenSize = 999;

            Assert.Equal(32, config.HiddenSize);
            Assert.Equal(999, clone.HiddenSize);
        }

        [Fact]
        public void TransformerModelFactory_Create_BuildsModelFromConfigOnly()
        {
            var config = ValidConfig();

            var model = TransformerModelFactory.Create(config);

            Assert.NotNull(model);
            Assert.IsType<Neuraval.Core.Models.TransformerModel>(model);
        }

        [Fact]
        public void TransformerModelFactory_Create_WithUnknownArchitecture_Throws()
        {
            var config = ValidConfig();
            config.Architecture = "qwen_like";

            Assert.Throws<NotSupportedException>(() => TransformerModelFactory.Create(config));
        }

        [Fact]
        public void ModelConfig_FromTransformerConfig_ValidatesUnderlyingConfig()
        {
            var config = ValidConfig();
            config.VocabSize = 0;
            var modelConfig = ModelConfig.FromTransformerConfig(config);

            Assert.Throws<ArgumentOutOfRangeException>(() => modelConfig.Validate());
        }
    }
}
