using Neuraval.Abstractions;

namespace Neuraval.Core.Models.Architectures
{
    public interface IDecoderArchitecture
    {
        ArchitectureId Id { get; }

        TransformerConfig Config { get; }

        ITrainableModel Build();
    }
}
