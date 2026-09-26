using System;
using System.Collections.Generic;

namespace Neuraval.Core.Models
{
    public sealed class ModelConfig
    {
        public string ModelType { get; set; } = "neuraval";
        public string SchemaVersion { get; set; } = "1.0";
        public string[] Architectures { get; set; } = Array.Empty<string>();

        public TransformerConfig Transformer { get; set; } = new TransformerConfig();

        public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(ModelType))
                throw new InvalidOperationException("ModelType es requerido");

            if (Transformer == null)
                throw new InvalidOperationException("Transformer es requerido");

            Transformer.Validate();
        }

        public static ModelConfig FromTransformerConfig(TransformerConfig transformerConfig, string modelType = "neuraval")
        {
            if (transformerConfig == null)
                throw new ArgumentNullException(nameof(transformerConfig));

            return new ModelConfig
            {
                ModelType = modelType,
                Architectures = new[] { transformerConfig.Architecture },
                Transformer = transformerConfig
            };
        }
    }
}
