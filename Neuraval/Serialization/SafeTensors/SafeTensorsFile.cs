using System.Collections.Generic;

namespace Neuraval.Core.Serialization.SafeTensors
{
    public sealed class SafeTensorsFile
    {
        public IReadOnlyList<SafeTensorsEntry> Tensors { get; }

        public IReadOnlyDictionary<string, string> Metadata { get; }

        public SafeTensorsFile(IReadOnlyList<SafeTensorsEntry> tensors, IReadOnlyDictionary<string, string> metadata)
        {
            Tensors = tensors;
            Metadata = metadata;
        }

        public SafeTensorsEntry? Find(string name)
        {
            foreach (var entry in Tensors)
            {
                if (entry.Name == name)
                    return entry;
            }

            return null;
        }
    }
}
