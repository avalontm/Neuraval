using System;
using System.Globalization;
using Neuraval.Core.Serialization.WeightLoading;

namespace Neuraval.Core.Serialization.Gguf
{
    public static class GgufTensorNameMapper
    {
        private const string TokenEmbedName = "token_embd.weight";
        private const string OutputNormName = "output_norm.weight";
        private const string OutputName = "output.weight";
        private const string BlockPrefix = "blk.";

        public static bool TryMapToNeuravalName(string ggufName, out string? neuravalName)
        {
            if (ggufName == null)
                throw new ArgumentNullException(nameof(ggufName));

            if (ggufName == TokenEmbedName)
            {
                neuravalName = TensorNameMapper.EmbedTokensName;
                return true;
            }

            if (ggufName == OutputNormName)
            {
                neuravalName = TensorNameMapper.FinalNormName;
                return true;
            }

            if (ggufName == OutputName)
            {
                neuravalName = TensorNameMapper.LmHeadName;
                return true;
            }

            if (TryParseBlockTensor(ggufName, out var layerIndex, out var suffix))
            {
                neuravalName = suffix switch
                {
                    "attn_norm.weight" => TensorNameMapper.InputLayerNormName(layerIndex),
                    "attn_q.weight" => TensorNameMapper.SelfAttnQProjName(layerIndex),
                    "attn_k.weight" => TensorNameMapper.SelfAttnKProjName(layerIndex),
                    "attn_v.weight" => TensorNameMapper.SelfAttnVProjName(layerIndex),
                    "attn_output.weight" => TensorNameMapper.SelfAttnOProjName(layerIndex),
                    "ffn_norm.weight" => TensorNameMapper.PostAttentionLayerNormName(layerIndex),
                    "ffn_gate.weight" => TensorNameMapper.MlpGateProjName(layerIndex),
                    "ffn_up.weight" => TensorNameMapper.MlpUpProjName(layerIndex),
                    "ffn_down.weight" => TensorNameMapper.MlpDownProjName(layerIndex),
                    _ => null
                };

                return neuravalName != null;
            }

            neuravalName = null;
            return false;
        }

        public static bool TryParseBlockTensor(string ggufName, out int layerIndex, out string suffix)
        {
            layerIndex = -1;
            suffix = string.Empty;

            if (!ggufName.StartsWith(BlockPrefix, StringComparison.Ordinal))
                return false;

            var remainder = ggufName.Substring(BlockPrefix.Length);
            var dotIndex = remainder.IndexOf('.');
            if (dotIndex <= 0)
                return false;

            var indexSegment = remainder.Substring(0, dotIndex);
            if (!int.TryParse(indexSegment, NumberStyles.None, CultureInfo.InvariantCulture, out layerIndex))
                return false;

            suffix = remainder.Substring(dotIndex + 1);
            return true;
        }
    }
}
