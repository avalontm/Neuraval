using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.Gguf;
using Neuraval.Core.Serialization.SafeTensors;
using Neuraval.Core.Serialization.WeightLoading;
using Xunit;

namespace Neuraval.Tests
{
    public class GgufModelWeightLoaderTests
    {
        private const int Hidden = 4;
        private const int Heads = 2;
        private const int Intermediate = 8;
        private const int Vocab = 6;

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
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["general.architecture"] = Str("llama"),
                ["llama.embedding_length"] = U32(Hidden),
                ["llama.block_count"] = U32(1),
                ["llama.attention.head_count"] = U32(Heads),
                ["llama.attention.head_count_kv"] = U32(Heads),
                ["llama.feed_forward_length"] = U32(Intermediate),
                ["llama.context_length"] = U32(2048),
                ["llama.vocab_size"] = U32(Vocab)
            };

            var tensors = new List<GgufTensorEntry>
            {
                new("token_embd.weight", GgmlType.F32, new[] { Vocab, Hidden }, Sequence(Vocab * Hidden)),
                new("blk.0.attn_norm.weight", GgmlType.F32, new[] { Hidden }, Sequence(Hidden)),
                new("blk.0.attn_q.weight", GgmlType.F32, new[] { Hidden, Hidden }, Sequence(Hidden * Hidden)),
                new("blk.0.attn_k.weight", GgmlType.F32, new[] { Hidden, Hidden }, Sequence(Hidden * Hidden)),
                new("blk.0.attn_v.weight", GgmlType.F32, new[] { Hidden, Hidden }, Sequence(Hidden * Hidden)),
                new("blk.0.attn_output.weight", GgmlType.F32, new[] { Hidden, Hidden }, Sequence(Hidden * Hidden)),
                new("blk.0.ffn_norm.weight", GgmlType.F32, new[] { Hidden }, Sequence(Hidden)),
                new("blk.0.ffn_gate.weight", GgmlType.F32, new[] { Intermediate, Hidden }, Sequence(Intermediate * Hidden)),
                new("blk.0.ffn_up.weight", GgmlType.F32, new[] { Intermediate, Hidden }, Sequence(Intermediate * Hidden)),
                new("blk.0.ffn_down.weight", GgmlType.F32, new[] { Hidden, Intermediate }, Sequence(Hidden * Intermediate)),
                new("output_norm.weight", GgmlType.F32, new[] { Hidden }, Sequence(Hidden))
            };

            if (includeOutputTensor)
                tensors.Add(new GgufTensorEntry("output.weight", GgmlType.F32, new[] { Vocab, Hidden }, Sequence(Vocab * Hidden)));

            return new GgufFile(3, metadata, tensors);
        }

        [Fact]
        public void Load_TiedEmbeddings_ProducesUsableModel()
        {
            var file = BuildSingleLayerFile(includeOutputTensor: false);

            var model = GgufModelWeightLoader.Load(file);

            Assert.Equal(Vocab, model.VocabSize);
            Assert.Equal(Hidden, model.HiddenSize);
            Assert.True(model.TiesWordEmbeddings);

            var logits = model.Forward(new[] { 0, 1, 2 });

            Assert.Equal(3, logits.GetLength(0));
            Assert.Equal(Vocab, logits.GetLength(1));
            AssertNoNaNs(logits);
        }

        [Fact]
        public void Load_UntiedEmbeddings_ProducesCorrectOutputShape()
        {
            var file = BuildSingleLayerFile(includeOutputTensor: true);

            var model = GgufModelWeightLoader.Load(file);

            Assert.False(model.TiesWordEmbeddings);

            var logits = model.Forward(new[] { 0, 1 });

            Assert.Equal(2, logits.GetLength(0));
            Assert.Equal(Vocab, logits.GetLength(1));
            AssertNoNaNs(logits);
        }

        [Fact]
        public void Load_MissingTensor_ThrowsWithTensorNameListed()
        {
            var file = BuildSingleLayerFile(includeOutputTensor: false);
            var tensorsWithoutQProj = file.Tensors.Where(t => t.Name != "blk.0.attn_q.weight").ToList();
            var incompleteFile = new GgufFile(file.Version, file.Metadata, tensorsWithoutQProj);

            var exception = Assert.Throws<InvalidOperationException>(() => GgufModelWeightLoader.Load(incompleteFile));

            Assert.Contains(TensorNameMapper.SelfAttnQProjName(0), exception.Message);
        }

        [Fact]
        public void BuildState_MissingTensorInWeightsList_ThrowsWithTensorNameListed()
        {
            var config = new TransformerConfig
            {
                Architecture = "llama",
                VocabSize = Vocab,
                HiddenSize = Hidden,
                NumHiddenLayers = 1,
                NumAttentionHeads = Heads,
                NumKeyValueHeads = Heads,
                IntermediateSize = Intermediate,
                MaxPositionEmbeddings = 2048,
                TieWordEmbeddings = true
            };

            var weights = new List<SafeTensorsEntry>
            {
                new(TensorNameMapper.EmbedTokensName, SafeTensorsDType.F32, new[] { Vocab, Hidden }, Sequence(Vocab * Hidden)),
                new(TensorNameMapper.FinalNormName, SafeTensorsDType.F32, new[] { Hidden }, Sequence(Hidden))
            };

            var exception = Assert.Throws<InvalidOperationException>(() => GgufModelWeightLoader.BuildState(config, weights));

            Assert.Contains(TensorNameMapper.InputLayerNormName(0), exception.Message);
        }

        [Fact]
        public void BuildState_TransposesLinearWeightsFromOutInToInOut()
        {
            const int hidden = 2;
            const int heads = 1;
            const int intermediate = 2;
            const int vocab = 2;

            var config = new TransformerConfig
            {
                Architecture = "llama",
                VocabSize = vocab,
                HiddenSize = hidden,
                NumHiddenLayers = 1,
                NumAttentionHeads = heads,
                NumKeyValueHeads = heads,
                IntermediateSize = intermediate,
                MaxPositionEmbeddings = 2048,
                TieWordEmbeddings = true
            };

            var wqOutIn = new float[] { 10f, 20f, 30f, 40f };

            var weights = new List<SafeTensorsEntry>
            {
                new(TensorNameMapper.EmbedTokensName, SafeTensorsDType.F32, new[] { vocab, hidden }, Sequence(vocab * hidden)),
                new(TensorNameMapper.FinalNormName, SafeTensorsDType.F32, new[] { hidden }, Sequence(hidden)),
                new(TensorNameMapper.InputLayerNormName(0), SafeTensorsDType.F32, new[] { hidden }, Sequence(hidden)),
                new(TensorNameMapper.PostAttentionLayerNormName(0), SafeTensorsDType.F32, new[] { hidden }, Sequence(hidden)),
                new(TensorNameMapper.SelfAttnQProjName(0), SafeTensorsDType.F32, new[] { hidden, hidden }, wqOutIn),
                new(TensorNameMapper.SelfAttnKProjName(0), SafeTensorsDType.F32, new[] { hidden, hidden }, Sequence(hidden * hidden)),
                new(TensorNameMapper.SelfAttnVProjName(0), SafeTensorsDType.F32, new[] { hidden, hidden }, Sequence(hidden * hidden)),
                new(TensorNameMapper.SelfAttnOProjName(0), SafeTensorsDType.F32, new[] { hidden, hidden }, Sequence(hidden * hidden)),
                new(TensorNameMapper.MlpGateProjName(0), SafeTensorsDType.F32, new[] { intermediate, hidden }, Sequence(intermediate * hidden)),
                new(TensorNameMapper.MlpUpProjName(0), SafeTensorsDType.F32, new[] { intermediate, hidden }, Sequence(intermediate * hidden)),
                new(TensorNameMapper.MlpDownProjName(0), SafeTensorsDType.F32, new[] { hidden, intermediate }, Sequence(hidden * intermediate))
            };

            var state = GgufModelWeightLoader.BuildState(config, weights);
            var wq = state.BlockStates[0].AttentionState.Wq;

            Assert.Equal(10f, wq[0, 0]);
            Assert.Equal(30f, wq[0, 1]);
            Assert.Equal(20f, wq[1, 0]);
            Assert.Equal(40f, wq[1, 1]);
        }

        [Fact]
        public void BuildState_WrongTensorShape_ThrowsInvalidOperationException()
        {
            var config = new TransformerConfig
            {
                Architecture = "llama",
                VocabSize = Vocab,
                HiddenSize = Hidden,
                NumHiddenLayers = 1,
                NumAttentionHeads = Heads,
                NumKeyValueHeads = Heads,
                IntermediateSize = Intermediate,
                MaxPositionEmbeddings = 2048,
                TieWordEmbeddings = true
            };

            var weights = new List<SafeTensorsEntry>
            {
                new(TensorNameMapper.EmbedTokensName, SafeTensorsDType.F32, new[] { Vocab, Hidden + 1 }, Sequence(Vocab * (Hidden + 1))),
                new(TensorNameMapper.FinalNormName, SafeTensorsDType.F32, new[] { Hidden }, Sequence(Hidden))
            };

            Assert.Throws<InvalidOperationException>(() => GgufModelWeightLoader.BuildState(config, weights));
        }

        private static void AssertNoNaNs(float[,] matrix)
        {
            foreach (var value in matrix)
            {
                Assert.False(float.IsNaN(value));
                Assert.False(float.IsInfinity(value));
            }
        }
    }
}
