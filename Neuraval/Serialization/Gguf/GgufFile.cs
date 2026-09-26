using System.Collections.Generic;

namespace Neuraval.Core.Serialization.Gguf
{
    public sealed class GgufFile
    {
        public uint Version { get; }

        public IReadOnlyDictionary<string, GgufMetadataValue> Metadata { get; }

        public IReadOnlyList<GgufTensorEntry> Tensors { get; }

        public GgufFile(uint version, IReadOnlyDictionary<string, GgufMetadataValue> metadata, IReadOnlyList<GgufTensorEntry> tensors)
        {
            Version = version;
            Metadata = metadata;
            Tensors = tensors;
        }

        public GgufTensorEntry? Find(string name)
        {
            foreach (var tensor in Tensors)
            {
                if (tensor.Name == name)
                    return tensor;
            }

            return null;
        }
    }
}
