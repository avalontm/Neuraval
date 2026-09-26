using System;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.ModelExport;
using Xunit;

namespace Neuraval.Tests
{
    public class ModelConfigJsonConverterTests
    {
        private static TransformerConfig BuildConfig()
        {
            return new TransformerConfig
            {
                Architecture = "neuraval-decoder",
                VocabSize = 32000,
                HiddenSize = 256,
                NumHiddenLayers = 4,
                NumAttentionHeads = 8,
                NumKeyValueHeads = 4,
                IntermediateSize = 688,
                MaxPositionEmbeddings = 2048,
                RopeTheta = 1000000f,
                RmsNormEps = 1e-6f,
                Activation = "silu",
                NormType = "rmsnorm",
                AttentionBias = false,
                TieWordEmbeddings = true
            };
        }

        [Fact]
        public void ToJson_UsesSnakeCasePropertyNames()
        {
            var json = ModelConfigJsonConverter.ToJson(BuildConfig());

            Assert.Contains("\"vocab_size\"", json);
            Assert.Contains("\"num_hidden_layers\"", json);
            Assert.Contains("\"num_key_value_heads\"", json);
            Assert.Contains("\"rope_theta\"", json);
            Assert.Contains("\"rms_norm_eps\"", json);
            Assert.Contains("\"tie_word_embeddings\"", json);
        }

        [Fact]
        public void FromJson_ToJson_RoundTripsAllFields()
        {
            var original = BuildConfig();

            var json = ModelConfigJsonConverter.ToJson(original);
            var restored = ModelConfigJsonConverter.FromJson(json);

            Assert.Equal(original.Architecture, restored.Architecture);
            Assert.Equal(original.VocabSize, restored.VocabSize);
            Assert.Equal(original.HiddenSize, restored.HiddenSize);
            Assert.Equal(original.NumHiddenLayers, restored.NumHiddenLayers);
            Assert.Equal(original.NumAttentionHeads, restored.NumAttentionHeads);
            Assert.Equal(original.NumKeyValueHeads, restored.NumKeyValueHeads);
            Assert.Equal(original.IntermediateSize, restored.IntermediateSize);
            Assert.Equal(original.MaxPositionEmbeddings, restored.MaxPositionEmbeddings);
            Assert.Equal(original.RopeTheta, restored.RopeTheta);
            Assert.Equal(original.RmsNormEps, restored.RmsNormEps);
            Assert.Equal(original.Activation, restored.Activation);
            Assert.Equal(original.NormType, restored.NormType);
            Assert.Equal(original.AttentionBias, restored.AttentionBias);
            Assert.Equal(original.TieWordEmbeddings, restored.TieWordEmbeddings);
        }

        [Fact]
        public void ToJson_NullConfig_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => ModelConfigJsonConverter.ToJson(null!));
        }

        [Fact]
        public void FromJson_NullJson_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => ModelConfigJsonConverter.FromJson(null!));
        }

        [Fact]
        public void FromJson_InvalidJson_Throws()
        {
            Assert.ThrowsAny<Exception>(() => ModelConfigJsonConverter.FromJson("not json"));
        }
    }
}
