using System;
using Neuraval.Abstractions;

namespace Neuraval.Core.Models.Architectures
{
    public sealed class LegacyDecoderArchitecture : IDecoderArchitecture
    {
        public ArchitectureId Id => ArchitectureId.Legacy;

        public TransformerConfig Config { get; }

        public LegacyDecoderArchitecture(TransformerConfig config)
        {
            Config = config ?? throw new ArgumentNullException(nameof(config));
        }

        public ITrainableModel Build()
        {
            return new TransformerModel(
                Config.VocabSize,
                Config.HiddenSize,
                Config.NumHiddenLayers,
                Config.NumAttentionHeads,
                Config.IntermediateSize,
                Config.MaxPositionEmbeddings,
                Config.Dropout,
                Config.Seed);
        }
    }
}
