using System;
using System.Collections.Generic;
using System.Globalization;
using Neuraval.Core.Models;

namespace Neuraval.Core.Serialization.WeightLoading
{
    public static class TensorNameMapper
    {
        public const string EmbedTokensName = "model.embed_tokens.weight";
        public const string FinalNormName = "model.norm.weight";
        public const string LmHeadName = "lm_head.weight";

        private const string LayerPrefix = "model.layers.";

        public static string InputLayerNormName(int layerIndex)
        {
            EnsureNonNegative(layerIndex);
            return $"{LayerPrefix}{layerIndex}.input_layernorm.weight";
        }

        public static string PostAttentionLayerNormName(int layerIndex)
        {
            EnsureNonNegative(layerIndex);
            return $"{LayerPrefix}{layerIndex}.post_attention_layernorm.weight";
        }

        public static string SelfAttnQProjName(int layerIndex)
        {
            EnsureNonNegative(layerIndex);
            return $"{LayerPrefix}{layerIndex}.self_attn.q_proj.weight";
        }

        public static string SelfAttnQProjBiasName(int layerIndex) => $"{SelfAttnQProjName(layerIndex)}".Replace(".weight", ".bias", StringComparison.Ordinal);

        public static string SelfAttnKProjName(int layerIndex)
        {
            EnsureNonNegative(layerIndex);
            return $"{LayerPrefix}{layerIndex}.self_attn.k_proj.weight";
        }

        public static string SelfAttnKProjBiasName(int layerIndex) => $"{SelfAttnKProjName(layerIndex)}".Replace(".weight", ".bias", StringComparison.Ordinal);

        public static string SelfAttnVProjName(int layerIndex)
        {
            EnsureNonNegative(layerIndex);
            return $"{LayerPrefix}{layerIndex}.self_attn.v_proj.weight";
        }

        public static string SelfAttnVProjBiasName(int layerIndex) => $"{SelfAttnVProjName(layerIndex)}".Replace(".weight", ".bias", StringComparison.Ordinal);

        public static string SelfAttnOProjName(int layerIndex)
        {
            EnsureNonNegative(layerIndex);
            return $"{LayerPrefix}{layerIndex}.self_attn.o_proj.weight";
        }

        public static string MlpGateProjName(int layerIndex)
        {
            EnsureNonNegative(layerIndex);
            return $"{LayerPrefix}{layerIndex}.mlp.gate_proj.weight";
        }

        public static string MlpUpProjName(int layerIndex)
        {
            EnsureNonNegative(layerIndex);
            return $"{LayerPrefix}{layerIndex}.mlp.up_proj.weight";
        }

        public static string MlpDownProjName(int layerIndex)
        {
            EnsureNonNegative(layerIndex);
            return $"{LayerPrefix}{layerIndex}.mlp.down_proj.weight";
        }

        public static IReadOnlyList<string> AllExpectedNames(TransformerConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            config.Validate();

            var names = new List<string> { EmbedTokensName };

            for (int layerIndex = 0; layerIndex < config.NumHiddenLayers; layerIndex++)
            {
                names.Add(InputLayerNormName(layerIndex));
                names.Add(SelfAttnQProjName(layerIndex));
                names.Add(SelfAttnKProjName(layerIndex));
                names.Add(SelfAttnVProjName(layerIndex));
                names.Add(SelfAttnOProjName(layerIndex));
                names.Add(PostAttentionLayerNormName(layerIndex));
                names.Add(MlpGateProjName(layerIndex));
                names.Add(MlpUpProjName(layerIndex));
                names.Add(MlpDownProjName(layerIndex));
            }

            names.Add(FinalNormName);

            if (!config.TieWordEmbeddings)
                names.Add(LmHeadName);

            return names;
        }

        public static int? TryParseLayerIndex(string tensorName)
        {
            if (tensorName == null)
                throw new ArgumentNullException(nameof(tensorName));

            if (!tensorName.StartsWith(LayerPrefix, StringComparison.Ordinal))
                return null;

            var remainder = tensorName.Substring(LayerPrefix.Length);
            var dotIndex = remainder.IndexOf('.');
            if (dotIndex <= 0)
                return null;

            var indexSegment = remainder.Substring(0, dotIndex);
            if (!int.TryParse(indexSegment, NumberStyles.None, CultureInfo.InvariantCulture, out var layerIndex))
                return null;

            return layerIndex;
        }

        public static bool TryGetRole(string tensorName, out TensorRole role, out int layerIndex)
        {
            if (tensorName == null)
                throw new ArgumentNullException(nameof(tensorName));

            role = default;
            layerIndex = -1;

            if (tensorName == EmbedTokensName)
            {
                role = TensorRole.EmbedTokens;
                return true;
            }

            if (tensorName == FinalNormName)
            {
                role = TensorRole.FinalNorm;
                return true;
            }

            if (tensorName == LmHeadName)
            {
                role = TensorRole.LmHead;
                return true;
            }

            var parsedLayerIndex = TryParseLayerIndex(tensorName);
            if (parsedLayerIndex == null)
                return false;

            layerIndex = parsedLayerIndex.Value;

            if (tensorName == InputLayerNormName(layerIndex))
                role = TensorRole.InputLayerNorm;
            else if (tensorName == PostAttentionLayerNormName(layerIndex))
                role = TensorRole.PostAttentionLayerNorm;
            else if (tensorName == SelfAttnQProjName(layerIndex))
                role = TensorRole.SelfAttnQProj;
            else if (tensorName == SelfAttnKProjName(layerIndex))
                role = TensorRole.SelfAttnKProj;
            else if (tensorName == SelfAttnVProjName(layerIndex))
                role = TensorRole.SelfAttnVProj;
            else if (tensorName == SelfAttnOProjName(layerIndex))
                role = TensorRole.SelfAttnOProj;
            else if (tensorName == MlpGateProjName(layerIndex))
                role = TensorRole.MlpGateProj;
            else if (tensorName == MlpUpProjName(layerIndex))
                role = TensorRole.MlpUpProj;
            else if (tensorName == MlpDownProjName(layerIndex))
                role = TensorRole.MlpDownProj;
            else
            {
                layerIndex = -1;
                return false;
            }

            return true;
        }

        private static void EnsureNonNegative(int layerIndex)
        {
            if (layerIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(layerIndex));
        }
    }
}
