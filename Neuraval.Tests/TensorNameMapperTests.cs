using System;
using System.Linq;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.WeightLoading;
using Xunit;

namespace Neuraval.Tests
{
    public class TensorNameMapperTests
    {
        private static TransformerConfig BuildConfig(int numHiddenLayers = 2, bool tieWordEmbeddings = false)
        {
            return new TransformerConfig
            {
                VocabSize = 64,
                HiddenSize = 16,
                NumHiddenLayers = numHiddenLayers,
                NumAttentionHeads = 4,
                NumKeyValueHeads = 2,
                IntermediateSize = 32,
                MaxPositionEmbeddings = 128,
                TieWordEmbeddings = tieWordEmbeddings
            };
        }

        [Fact]
        public void EmbedTokensName_MatchesRoadmapConvention()
        {
            Assert.Equal("model.embed_tokens.weight", TensorNameMapper.EmbedTokensName);
        }

        [Fact]
        public void FinalNormName_MatchesRoadmapConvention()
        {
            Assert.Equal("model.norm.weight", TensorNameMapper.FinalNormName);
        }

        [Fact]
        public void LmHeadName_MatchesRoadmapConvention()
        {
            Assert.Equal("lm_head.weight", TensorNameMapper.LmHeadName);
        }

        [Fact]
        public void PerLayerNames_MatchRoadmapConvention()
        {
            Assert.Equal("model.layers.0.input_layernorm.weight", TensorNameMapper.InputLayerNormName(0));
            Assert.Equal("model.layers.0.post_attention_layernorm.weight", TensorNameMapper.PostAttentionLayerNormName(0));
            Assert.Equal("model.layers.3.self_attn.q_proj.weight", TensorNameMapper.SelfAttnQProjName(3));
            Assert.Equal("model.layers.3.self_attn.k_proj.weight", TensorNameMapper.SelfAttnKProjName(3));
            Assert.Equal("model.layers.3.self_attn.v_proj.weight", TensorNameMapper.SelfAttnVProjName(3));
            Assert.Equal("model.layers.3.self_attn.o_proj.weight", TensorNameMapper.SelfAttnOProjName(3));
            Assert.Equal("model.layers.3.mlp.gate_proj.weight", TensorNameMapper.MlpGateProjName(3));
            Assert.Equal("model.layers.3.mlp.up_proj.weight", TensorNameMapper.MlpUpProjName(3));
            Assert.Equal("model.layers.3.mlp.down_proj.weight", TensorNameMapper.MlpDownProjName(3));
        }

        [Theory]
        [InlineData(-1)]
        [InlineData(-5)]
        public void PerLayerNames_NegativeLayerIndex_Throws(int layerIndex)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => TensorNameMapper.InputLayerNormName(layerIndex));
        }

        [Fact]
        public void AllExpectedNames_UntiedEmbeddings_IncludesLmHeadAndAllLayers()
        {
            var names = TensorNameMapper.AllExpectedNames(BuildConfig(numHiddenLayers: 2, tieWordEmbeddings: false));

            Assert.Equal(1 + 2 * 9 + 1 + 1, names.Count);
            Assert.Contains("model.embed_tokens.weight", names);
            Assert.Contains("model.norm.weight", names);
            Assert.Contains("lm_head.weight", names);
            Assert.Contains("model.layers.1.mlp.down_proj.weight", names);
        }

        [Fact]
        public void AllExpectedNames_TiedEmbeddings_ExcludesLmHead()
        {
            var names = TensorNameMapper.AllExpectedNames(BuildConfig(numHiddenLayers: 1, tieWordEmbeddings: true));

            Assert.DoesNotContain("lm_head.weight", names);
            Assert.Equal(1 + 1 * 9 + 1, names.Count);
        }

        [Fact]
        public void AllExpectedNames_NullConfig_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => TensorNameMapper.AllExpectedNames(null!));
        }

        [Fact]
        public void AllExpectedNames_NamesAreUnique()
        {
            var names = TensorNameMapper.AllExpectedNames(BuildConfig(numHiddenLayers: 4));

            Assert.Equal(names.Count, names.Distinct().Count());
        }

        [Theory]
        [InlineData("model.layers.0.self_attn.q_proj.weight", 0)]
        [InlineData("model.layers.12.mlp.down_proj.weight", 12)]
        public void TryParseLayerIndex_ValidNames_ReturnsIndex(string tensorName, int expected)
        {
            Assert.Equal(expected, TensorNameMapper.TryParseLayerIndex(tensorName));
        }

        [Theory]
        [InlineData("model.embed_tokens.weight")]
        [InlineData("lm_head.weight")]
        [InlineData("model.layers.notanumber.mlp.down_proj.weight")]
        [InlineData("model.layers.")]
        public void TryParseLayerIndex_InvalidNames_ReturnsNull(string tensorName)
        {
            Assert.Null(TensorNameMapper.TryParseLayerIndex(tensorName));
        }

        [Fact]
        public void TryGetRole_EmbedTokens_ReturnsRoleWithoutLayerIndex()
        {
            var found = TensorNameMapper.TryGetRole("model.embed_tokens.weight", out var role, out var layerIndex);

            Assert.True(found);
            Assert.Equal(TensorRole.EmbedTokens, role);
            Assert.Equal(-1, layerIndex);
        }

        [Fact]
        public void TryGetRole_LayeredTensor_ReturnsRoleAndLayerIndex()
        {
            var found = TensorNameMapper.TryGetRole("model.layers.2.self_attn.v_proj.weight", out var role, out var layerIndex);

            Assert.True(found);
            Assert.Equal(TensorRole.SelfAttnVProj, role);
            Assert.Equal(2, layerIndex);
        }

        [Fact]
        public void TryGetRole_UnknownName_ReturnsFalse()
        {
            var found = TensorNameMapper.TryGetRole("model.layers.0.unknown.weight", out _, out var layerIndex);

            Assert.False(found);
            Assert.Equal(-1, layerIndex);
        }
    }
}
