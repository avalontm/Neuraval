using Neuraval.Core.Serialization.Gguf;
using Neuraval.Core.Serialization.WeightLoading;
using Xunit;

namespace Neuraval.Tests
{
    public class GgufTensorNameMapperTests
    {
        [Fact]
        public void TryMapToNeuravalName_TokenEmbed_MapsToEmbedTokens()
        {
            Assert.True(GgufTensorNameMapper.TryMapToNeuravalName("token_embd.weight", out var name));
            Assert.Equal(TensorNameMapper.EmbedTokensName, name);
        }

        [Fact]
        public void TryMapToNeuravalName_OutputNorm_MapsToFinalNorm()
        {
            Assert.True(GgufTensorNameMapper.TryMapToNeuravalName("output_norm.weight", out var name));
            Assert.Equal(TensorNameMapper.FinalNormName, name);
        }

        [Fact]
        public void TryMapToNeuravalName_Output_MapsToLmHead()
        {
            Assert.True(GgufTensorNameMapper.TryMapToNeuravalName("output.weight", out var name));
            Assert.Equal(TensorNameMapper.LmHeadName, name);
        }

        [Theory]
        [InlineData("attn_norm.weight")]
        [InlineData("attn_q.weight")]
        [InlineData("attn_q.bias")]
        [InlineData("attn_k.weight")]
        [InlineData("attn_k.bias")]
        [InlineData("attn_v.weight")]
        [InlineData("attn_v.bias")]
        [InlineData("attn_output.weight")]
        [InlineData("ffn_norm.weight")]
        [InlineData("ffn_gate.weight")]
        [InlineData("ffn_up.weight")]
        [InlineData("ffn_down.weight")]
        public void TryMapToNeuravalName_BlockTensors_MapToLayerIndexedNames(string suffix)
        {
            Assert.True(GgufTensorNameMapper.TryMapToNeuravalName($"blk.3.{suffix}", out var name));
            Assert.Contains("3", name);
        }

        [Fact]
        public void TryMapToNeuravalName_UnknownTensor_ReturnsFalse()
        {
            Assert.False(GgufTensorNameMapper.TryMapToNeuravalName("rope_freqs.weight", out var name));
            Assert.Null(name);
        }

        [Fact]
        public void TryMapToNeuravalName_UnknownBlockSuffix_ReturnsFalse()
        {
            Assert.False(GgufTensorNameMapper.TryMapToNeuravalName("blk.0.attn_q_norm.weight", out var name));
            Assert.Null(name);
        }

        [Fact]
        public void TryParseBlockTensor_ValidName_ExtractsLayerAndSuffix()
        {
            Assert.True(GgufTensorNameMapper.TryParseBlockTensor("blk.12.ffn_down.weight", out var layerIndex, out var suffix));
            Assert.Equal(12, layerIndex);
            Assert.Equal("ffn_down.weight", suffix);
        }

        [Fact]
        public void TryParseBlockTensor_NonBlockName_ReturnsFalse()
        {
            Assert.False(GgufTensorNameMapper.TryParseBlockTensor("output.weight", out _, out _));
        }
    }
}
