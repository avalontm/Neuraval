using System;
using System.Collections.Generic;
using Neuraval.Core.Serialization.Gguf;
using Xunit;

namespace Neuraval.Tests
{
    public class GgufConfigMapperTests
    {
        private static GgufMetadataValue U32(uint value) => new(GgufValueType.UInt32, value);
        private static GgufMetadataValue F32(float value) => new(GgufValueType.Float32, value);
        private static GgufMetadataValue Str(string value) => new(GgufValueType.String, value);

        private static float[] Zeros(int count) => new float[count];

        [Fact]
        public void MapConfig_WithExplicitMetadata_ResolvesAllFields()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["general.architecture"] = Str("llama"),
                ["llama.embedding_length"] = U32(64),
                ["llama.block_count"] = U32(2),
                ["llama.attention.head_count"] = U32(8),
                ["llama.attention.head_count_kv"] = U32(2),
                ["llama.feed_forward_length"] = U32(256),
                ["llama.context_length"] = U32(2048),
                ["llama.vocab_size"] = U32(1000),
                ["llama.rope.freq_base"] = F32(1000000f),
                ["llama.attention.layer_norm_rms_epsilon"] = F32(1e-5f)
            };

            var file = new GgufFile(3, metadata, Array.Empty<GgufTensorEntry>());
            var config = GgufConfigMapper.MapConfig(file);

            Assert.Equal("llama", config.Architecture);
            Assert.Equal(64, config.HiddenSize);
            Assert.Equal(2, config.NumHiddenLayers);
            Assert.Equal(8, config.NumAttentionHeads);
            Assert.Equal(2, config.NumKeyValueHeads);
            Assert.Equal(256, config.IntermediateSize);
            Assert.Equal(2048, config.MaxPositionEmbeddings);
            Assert.Equal(1000, config.VocabSize);
            Assert.Equal(1000000f, config.RopeTheta);
            Assert.Equal(1e-5f, config.RmsNormEps);
            Assert.True(config.TieWordEmbeddings);
        }

        [Fact]
        public void MapConfig_WithoutHeadCountKv_DefaultsToMultiHeadAttention()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["general.architecture"] = Str("llama"),
                ["llama.embedding_length"] = U32(64),
                ["llama.block_count"] = U32(1),
                ["llama.attention.head_count"] = U32(8),
                ["llama.feed_forward_length"] = U32(128),
                ["llama.context_length"] = U32(2048),
                ["llama.vocab_size"] = U32(100)
            };

            var file = new GgufFile(3, metadata, Array.Empty<GgufTensorEntry>());
            var config = GgufConfigMapper.MapConfig(file);

            Assert.Equal(8, config.NumKeyValueHeads);
        }

        [Fact]
        public void MapConfig_WithOutputTensorPresent_TieWordEmbeddingsIsFalse()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["general.architecture"] = Str("llama"),
                ["llama.embedding_length"] = U32(4),
                ["llama.block_count"] = U32(1),
                ["llama.attention.head_count"] = U32(2),
                ["llama.feed_forward_length"] = U32(8),
                ["llama.context_length"] = U32(2048),
                ["llama.vocab_size"] = U32(10)
            };

            var tensors = new[]
            {
                new GgufTensorEntry("output.weight", GgmlType.F32, new[] { 10, 4 }, Zeros(40))
            };

            var file = new GgufFile(3, metadata, tensors);
            var config = GgufConfigMapper.MapConfig(file);

            Assert.False(config.TieWordEmbeddings);
        }

        [Fact]
        public void MapConfig_MissingBlockCount_InfersFromTensorNames()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["general.architecture"] = Str("llama"),
                ["llama.embedding_length"] = U32(4),
                ["llama.attention.head_count"] = U32(2),
                ["llama.feed_forward_length"] = U32(8),
                ["llama.context_length"] = U32(2048),
                ["llama.vocab_size"] = U32(10)
            };

            var tensors = new[]
            {
                new GgufTensorEntry("blk.0.attn_q.weight", GgmlType.F32, new[] { 4, 4 }, Zeros(16)),
                new GgufTensorEntry("blk.1.attn_q.weight", GgmlType.F32, new[] { 4, 4 }, Zeros(16)),
                new GgufTensorEntry("blk.2.attn_q.weight", GgmlType.F32, new[] { 4, 4 }, Zeros(16))
            };

            var file = new GgufFile(3, metadata, tensors);
            var config = GgufConfigMapper.MapConfig(file);

            Assert.Equal(3, config.NumHiddenLayers);
        }

        [Fact]
        public void MapConfig_MissingVocabSize_InfersFromEmbeddingTensorShape()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["general.architecture"] = Str("llama"),
                ["llama.embedding_length"] = U32(4),
                ["llama.block_count"] = U32(1),
                ["llama.attention.head_count"] = U32(2),
                ["llama.feed_forward_length"] = U32(8),
                ["llama.context_length"] = U32(2048)
            };

            var tensors = new[]
            {
                new GgufTensorEntry("token_embd.weight", GgmlType.F32, new[] { 500, 4 }, Zeros(2000))
            };

            var file = new GgufFile(3, metadata, tensors);
            var config = GgufConfigMapper.MapConfig(file);

            Assert.Equal(500, config.VocabSize);
        }

        [Fact]
        public void MapConfig_MissingArchitecture_Throws()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>();
            var file = new GgufFile(3, metadata, Array.Empty<GgufTensorEntry>());

            Assert.Throws<InvalidOperationException>(() => GgufConfigMapper.MapConfig(file));
        }

        [Fact]
        public void MapConfig_MissingContextLength_DefaultsTo2048()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["general.architecture"] = Str("llama"),
                ["llama.embedding_length"] = U32(4),
                ["llama.block_count"] = U32(1),
                ["llama.attention.head_count"] = U32(2),
                ["llama.feed_forward_length"] = U32(8),
                ["llama.vocab_size"] = U32(10)
            };

            var file = new GgufFile(3, metadata, Array.Empty<GgufTensorEntry>());
            var config = GgufConfigMapper.MapConfig(file);

            Assert.Equal(2048, config.MaxPositionEmbeddings);
        }
    }
}
