using System;
using Neuraval.Core.Models;

namespace Neuraval.Core.Serialization.Gguf
{
    public static class GgufConfigMapper
    {
        public static TransformerConfig MapConfig(GgufFile file)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));

            var metadata = file.Metadata;

            if (!metadata.TryGetString("general.architecture", out var architecture) || string.IsNullOrEmpty(architecture))
                throw new InvalidOperationException("El archivo GGUF no declara 'general.architecture'");

            var config = new TransformerConfig
            {
                Architecture = architecture,
                HiddenSize = ResolveHiddenSize(file, architecture),
                NumHiddenLayers = ResolveNumHiddenLayers(file, architecture),
                NumAttentionHeads = ResolveNumAttentionHeads(file, architecture),
                IntermediateSize = ResolveIntermediateSize(file, architecture),
                MaxPositionEmbeddings = ResolveContextLength(file, architecture),
                VocabSize = ResolveVocabSize(file, architecture),
                RopeTheta = metadata.TryGetFloat($"{architecture}.rope.freq_base", out var ropeTheta) ? ropeTheta : 10000f,
                RmsNormEps = metadata.TryGetFloat($"{architecture}.attention.layer_norm_rms_epsilon", out var eps) ? eps : 1e-6f,
                Activation = "silu",
                NormType = "rmsnorm",
                AttentionBias = file.Find("blk.0.attn_q.bias") != null,
                TieWordEmbeddings = file.Find("output.weight") == null
            };

            config.NumKeyValueHeads = metadata.TryGetInt32($"{architecture}.attention.head_count_kv", out var kvHeads)
                ? kvHeads
                : config.NumAttentionHeads;

            return config;
        }

        private static int ResolveHiddenSize(GgufFile file, string architecture)
        {
            if (file.Metadata.TryGetInt32($"{architecture}.embedding_length", out var value))
                return value;

            var embedTokens = file.Find("token_embd.weight");
            if (embedTokens != null && embedTokens.Shape.Length == 2)
                return embedTokens.Shape[1];

            throw new InvalidOperationException($"No se pudo resolver hidden_size para la arquitectura '{architecture}'");
        }

        private static int ResolveNumHiddenLayers(GgufFile file, string architecture)
        {
            if (file.Metadata.TryGetInt32($"{architecture}.block_count", out var value))
                return value;

            int maxLayerIndex = -1;
            foreach (var tensor in file.Tensors)
            {
                if (GgufTensorNameMapper.TryParseBlockTensor(tensor.Name, out var layerIndex, out _) && layerIndex > maxLayerIndex)
                    maxLayerIndex = layerIndex;
            }

            if (maxLayerIndex >= 0)
                return maxLayerIndex + 1;

            throw new InvalidOperationException($"No se pudo resolver num_hidden_layers para la arquitectura '{architecture}'");
        }

        private static int ResolveNumAttentionHeads(GgufFile file, string architecture)
        {
            if (file.Metadata.TryGetInt32($"{architecture}.attention.head_count", out var value))
                return value;

            throw new InvalidOperationException($"No se pudo resolver num_attention_heads para la arquitectura '{architecture}'");
        }

        private static int ResolveIntermediateSize(GgufFile file, string architecture)
        {
            if (file.Metadata.TryGetInt32($"{architecture}.feed_forward_length", out var value))
                return value;

            var gateProj = file.Find("blk.0.ffn_gate.weight");
            if (gateProj != null && gateProj.Shape.Length == 2)
                return gateProj.Shape[0];

            throw new InvalidOperationException($"No se pudo resolver intermediate_size para la arquitectura '{architecture}'");
        }

        private static int ResolveContextLength(GgufFile file, string architecture)
        {
            if (file.Metadata.TryGetInt32($"{architecture}.context_length", out var value))
                return value;

            return 2048;
        }

        private static int ResolveVocabSize(GgufFile file, string architecture)
        {
            if (file.Metadata.TryGetInt32($"{architecture}.vocab_size", out var value))
                return value;

            if (file.Metadata.TryGetArrayLength("tokenizer.ggml.tokens", out var tokenCount) && tokenCount > 0)
                return tokenCount;

            var embedTokens = file.Find("token_embd.weight");
            if (embedTokens != null && embedTokens.Shape.Length == 2)
                return embedTokens.Shape[0];

            throw new InvalidOperationException($"No se pudo resolver vocab_size para la arquitectura '{architecture}'");
        }
    }
}
