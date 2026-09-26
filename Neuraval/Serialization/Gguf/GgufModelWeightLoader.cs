using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.SafeTensors;
using Neuraval.Core.Serialization.WeightLoading;

namespace Neuraval.Core.Serialization.Gguf
{
    public static class GgufModelWeightLoader
    {
        public static ModernDecoderModel Load(string filePath)
        {
            return Load(GgufModelLoader.Load(filePath));
        }

        public static ModernDecoderModel Load(GgufFile file)
        {
            return Load(GgufModelLoader.Load(file));
        }

        public static ModernDecoderModel Load(GgufLoadResult result)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (!result.Report.IsComplete)
                throw new InvalidOperationException(
                    $"El GGUF no contiene todos los tensores requeridos por la configuración resuelta. Faltan: {string.Join(", ", result.Report.MissingNames)}");

            var state = BuildState(result.Config, result.Weights);
            return ModernDecoderModel.LoadState(state);
        }

        public static ModernDecoderModelState BuildState(TransformerConfig config, IReadOnlyList<SafeTensorsEntry> weights)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            if (weights == null)
                throw new ArgumentNullException(nameof(weights));

            config.Validate();

            var byName = weights.ToDictionary(entry => entry.Name, entry => entry);

            var state = new ModernDecoderModelState
            {
                Config = config.Clone(),
                EmbeddingState = BuildEmbeddingState(config, byName),
                FinalNormState = BuildNormState(config, RequireTensor(byName, TensorNameMapper.FinalNormName)),
                BlockStates = new List<ModernDecoderBlockState>()
            };

            for (int layerIndex = 0; layerIndex < config.NumHiddenLayers; layerIndex++)
                state.BlockStates.Add(BuildBlockState(config, layerIndex, byName));

            if (!config.TieWordEmbeddings)
            {
                var lmHead = RequireTensor(byName, TensorNameMapper.LmHeadName);
                ValidateShape(lmHead, config.VocabSize, config.HiddenSize);
                state.OutputWeights = (float[])lmHead.Data.Clone();
            }

            return state;
        }

        private static EmbeddingLayerState BuildEmbeddingState(TransformerConfig config, Dictionary<string, SafeTensorsEntry> byName)
        {
            var embedTokens = RequireTensor(byName, TensorNameMapper.EmbedTokensName);
            ValidateShape(embedTokens, config.VocabSize, config.HiddenSize);

            return new EmbeddingLayerState
            {
                VocabSize = config.VocabSize,
                EmbeddingDim = config.HiddenSize,
                Embeddings = (float[])embedTokens.Data.Clone()
            };
        }

        private static RMSNormState BuildNormState(TransformerConfig config, SafeTensorsEntry entry)
        {
            ValidateShape1D(entry, config.HiddenSize);

            return new RMSNormState
            {
                NormalizedShape = config.HiddenSize,
                Epsilon = config.RmsNormEps,
                Weight = (float[])entry.Data.Clone()
            };
        }

        private static ModernDecoderBlockState BuildBlockState(TransformerConfig config, int layerIndex, Dictionary<string, SafeTensorsEntry> byName)
        {
            int hidden = config.HiddenSize;
            int kvDim = config.NumKeyValueHeads * config.HeadDim;
            int intermediate = config.IntermediateSize;

            var inputNorm = RequireTensor(byName, TensorNameMapper.InputLayerNormName(layerIndex));
            var postNorm = RequireTensor(byName, TensorNameMapper.PostAttentionLayerNormName(layerIndex));

            var wq = RequireTensor(byName, TensorNameMapper.SelfAttnQProjName(layerIndex));
            var wk = RequireTensor(byName, TensorNameMapper.SelfAttnKProjName(layerIndex));
            var wv = RequireTensor(byName, TensorNameMapper.SelfAttnVProjName(layerIndex));
            var wo = RequireTensor(byName, TensorNameMapper.SelfAttnOProjName(layerIndex));

            var gate = RequireTensor(byName, TensorNameMapper.MlpGateProjName(layerIndex));
            var up = RequireTensor(byName, TensorNameMapper.MlpUpProjName(layerIndex));
            var down = RequireTensor(byName, TensorNameMapper.MlpDownProjName(layerIndex));

            ValidateShape(wq, hidden, hidden);
            ValidateShape(wk, kvDim, hidden);
            ValidateShape(wv, kvDim, hidden);
            ValidateShape(wo, hidden, hidden);
            ValidateShape(gate, intermediate, hidden);
            ValidateShape(up, intermediate, hidden);
            ValidateShape(down, hidden, intermediate);

            var attentionState = new GQAAttentionState
            {
                HiddenSize = hidden,
                NumAttentionHeads = config.NumAttentionHeads,
                NumKeyValueHeads = config.NumKeyValueHeads,
                Wq = ToInOutMatrix2D(wq.Data, hidden, hidden),
                Wk = ToInOutMatrix2D(wk.Data, kvDim, hidden),
                Wv = ToInOutMatrix2D(wv.Data, kvDim, hidden),
                Wo = ToInOutMatrix2D(wo.Data, hidden, hidden)
            };

            var feedforwardState = new SwiGLUFeedForwardState
            {
                EmbeddingDim = hidden,
                HiddenDim = intermediate,
                WeightsGate = ToInOutMatrixFlat(gate.Data, intermediate, hidden),
                WeightsUp = ToInOutMatrixFlat(up.Data, intermediate, hidden),
                WeightsDown = ToInOutMatrixFlat(down.Data, hidden, intermediate)
            };

            return new ModernDecoderBlockState
            {
                Config = config.Clone(),
                Seed = config.Seed,
                Norm1State = BuildNormState(config, inputNorm),
                Norm2State = BuildNormState(config, postNorm),
                AttentionState = attentionState,
                FeedforwardState = feedforwardState
            };
        }

        private static float[,] ToInOutMatrix2D(float[] data, int outDim, int inDim)
        {
            var result = new float[inDim, outDim];

            for (int o = 0; o < outDim; o++)
                for (int i = 0; i < inDim; i++)
                    result[i, o] = data[o * inDim + i];

            return result;
        }

        private static float[] ToInOutMatrixFlat(float[] data, int outDim, int inDim)
        {
            var result = new float[data.Length];

            for (int o = 0; o < outDim; o++)
                for (int i = 0; i < inDim; i++)
                    result[i * outDim + o] = data[o * inDim + i];

            return result;
        }

        private static SafeTensorsEntry RequireTensor(Dictionary<string, SafeTensorsEntry> byName, string name)
        {
            if (!byName.TryGetValue(name, out var entry))
                throw new InvalidOperationException($"Falta el tensor requerido '{name}' para construir el ModernDecoderModel");

            return entry;
        }

        private static void ValidateShape(SafeTensorsEntry entry, int expectedOutDim, int expectedInDim)
        {
            if (entry.Shape.Length != 2 || entry.Shape[0] != expectedOutDim || entry.Shape[1] != expectedInDim)
                throw new InvalidOperationException(
                    $"El tensor '{entry.Name}' tiene shape [{string.Join(",", entry.Shape)}] pero se esperaba [{expectedOutDim},{expectedInDim}]");
        }

        private static void ValidateShape1D(SafeTensorsEntry entry, int expectedDim)
        {
            if (entry.Shape.Length != 1 || entry.Shape[0] != expectedDim)
                throw new InvalidOperationException(
                    $"El tensor '{entry.Name}' tiene shape [{string.Join(",", entry.Shape)}] pero se esperaba [{expectedDim}]");
        }
    }
}
