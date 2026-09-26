using System;
using Neuraval.Abstractions;

namespace Neuraval.Core.Models.Architectures
{
    public static class TransformerModelFactory
    {
        public static ITrainableModel Create(TransformerConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            config.Validate();

            return ResolveArchitecture(config).Build();
        }

        public static IDecoderArchitecture ResolveArchitecture(TransformerConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            return config.Architecture switch
            {
                "legacy" => new LegacyDecoderArchitecture(config),
                _ => throw new NotSupportedException(
                    $"Arquitectura '{config.Architecture}' no soportada todavía. Disponible: 'legacy'.")
            };
        }
    }
}
