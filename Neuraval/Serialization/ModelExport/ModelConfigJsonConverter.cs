using System;
using System.Text.Json;
using Neuraval.Core.Models;

namespace Neuraval.Core.Serialization.ModelExport
{
    public static class ModelConfigJsonConverter
    {
        private static readonly JsonSerializerOptions SerializerOptions = new()
        {
            WriteIndented = true
        };

        public static string ToJson(TransformerConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            var document = new ModelConfigDocument
            {
                Architecture = config.Architecture,
                VocabSize = config.VocabSize,
                HiddenSize = config.HiddenSize,
                NumHiddenLayers = config.NumHiddenLayers,
                NumAttentionHeads = config.NumAttentionHeads,
                NumKeyValueHeads = config.NumKeyValueHeads,
                IntermediateSize = config.IntermediateSize,
                MaxPositionEmbeddings = config.MaxPositionEmbeddings,
                RopeTheta = config.RopeTheta,
                RmsNormEps = config.RmsNormEps,
                Activation = config.Activation,
                NormType = config.NormType,
                AttentionBias = config.AttentionBias,
                TieWordEmbeddings = config.TieWordEmbeddings
            };

            return JsonSerializer.Serialize(document, SerializerOptions);
        }

        public static TransformerConfig FromJson(string json)
        {
            if (json == null)
                throw new ArgumentNullException(nameof(json));

            var document = JsonSerializer.Deserialize<ModelConfigDocument>(json)
                ?? throw new InvalidOperationException("No se pudo deserializar config.json");

            return new TransformerConfig
            {
                Architecture = document.Architecture,
                VocabSize = document.VocabSize,
                HiddenSize = document.HiddenSize,
                NumHiddenLayers = document.NumHiddenLayers,
                NumAttentionHeads = document.NumAttentionHeads,
                NumKeyValueHeads = document.NumKeyValueHeads,
                IntermediateSize = document.IntermediateSize,
                MaxPositionEmbeddings = document.MaxPositionEmbeddings,
                RopeTheta = document.RopeTheta,
                RmsNormEps = document.RmsNormEps,
                Activation = document.Activation,
                NormType = document.NormType,
                AttentionBias = document.AttentionBias,
                TieWordEmbeddings = document.TieWordEmbeddings
            };
        }
    }
}
