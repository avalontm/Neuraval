using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Core.Serialization.Gguf;
using Neuraval.Core.Serialization.WeightLoading;
using Xunit;

namespace Neuraval.Tests
{
    public class GgufModelLoaderTests
    {
        private static GgufMetadataValue U32(uint value) => new(GgufValueType.UInt32, value);
        private static GgufMetadataValue Str(string value) => new(GgufValueType.String, value);

        private static float[] Sequence(int count)
        {
            var data = new float[count];
            for (int i = 0; i < count; i++)
                data[i] = i;
            return data;
        }

        private static GgufFile BuildSingleLayerFile(bool includeOutputTensor)
        {
            const int hidden = 4;
            const int heads = 2;
            const int intermediate = 8;
            const int vocab = 6;

            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["general.architecture"] = Str("llama"),
                ["llama.embedding_length"] = U32(hidden),
                ["llama.block_count"] = U32(1),
                ["llama.attention.head_count"] = U32(heads),
                ["llama.attention.head_count_kv"] = U32(heads),
                ["llama.feed_forward_length"] = U32(intermediate),
                ["llama.context_length"] = U32(2048),
                ["llama.vocab_size"] = U32(vocab)
            };

            var tensors = new List<GgufTensorEntry>
            {
                new("token_embd.weight", GgmlType.F32, new[] { vocab, hidden }, Sequence(vocab * hidden)),
                new("blk.0.attn_norm.weight", GgmlType.F32, new[] { hidden }, Sequence(hidden)),
                new("blk.0.attn_q.weight", GgmlType.F32, new[] { hidden, hidden }, Sequence(hidden * hidden)),
                new("blk.0.attn_k.weight", GgmlType.F32, new[] { hidden, hidden }, Sequence(hidden * hidden)),
                new("blk.0.attn_v.weight", GgmlType.F32, new[] { hidden, hidden }, Sequence(hidden * hidden)),
                new("blk.0.attn_output.weight", GgmlType.F32, new[] { hidden, hidden }, Sequence(hidden * hidden)),
                new("blk.0.ffn_norm.weight", GgmlType.F32, new[] { hidden }, Sequence(hidden)),
                new("blk.0.ffn_gate.weight", GgmlType.F32, new[] { intermediate, hidden }, Sequence(intermediate * hidden)),
                new("blk.0.ffn_up.weight", GgmlType.F32, new[] { intermediate, hidden }, Sequence(intermediate * hidden)),
                new("blk.0.ffn_down.weight", GgmlType.F32, new[] { hidden, intermediate }, Sequence(hidden * intermediate)),
                new("output_norm.weight", GgmlType.F32, new[] { hidden }, Sequence(hidden))
            };

            if (includeOutputTensor)
                tensors.Add(new GgufTensorEntry("output.weight", GgmlType.F32, new[] { vocab, hidden }, Sequence(vocab * hidden)));

            return new GgufFile(3, metadata, tensors);
        }

        [Fact]
        public void Load_TiedEmbeddings_ProducesCompleteManifest()
        {
            var file = BuildSingleLayerFile(includeOutputTensor: false);
            var result = GgufModelLoader.Load(file);

            Assert.True(result.Report.IsComplete);
            Assert.Empty(result.Report.UnexpectedNames);
            Assert.True(result.Config.TieWordEmbeddings);
        }

        [Fact]
        public void Load_UntiedEmbeddings_ProducesCompleteManifest()
        {
            var file = BuildSingleLayerFile(includeOutputTensor: true);
            var result = GgufModelLoader.Load(file);

            Assert.True(result.Report.IsComplete);
            Assert.False(result.Config.TieWordEmbeddings);
            Assert.Contains(result.Weights, w => w.Name == TensorNameMapper.LmHeadName);
        }

        [Fact]
        public void Load_PreservesTensorValues()
        {
            var file = BuildSingleLayerFile(includeOutputTensor: false);
            var result = GgufModelLoader.Load(file);

            var embed = result.Weights.Single(w => w.Name == TensorNameMapper.EmbedTokensName);
            Assert.Equal(Sequence(6 * 4), embed.Data);
        }

        [Fact]
        public void LoadWeights_SkipsUnmappedTensors()
        {
            var file = BuildSingleLayerFile(includeOutputTensor: false);
            var weights = GgufModelLoader.LoadWeights(file);

            Assert.All(weights, w => Assert.False(string.IsNullOrEmpty(w.Name)));
            Assert.Equal(11, weights.Count);
        }
    }
}
