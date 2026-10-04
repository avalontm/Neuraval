using System;

namespace Neuraval.Core.Serialization
{
    public class LoraBinaryHeader
    {
        public int EmbeddingDim { get; set; }

        public int NumLayers { get; set; }

        public int Rank { get; set; }

        public float Alpha { get; set; }

        public bool FreezeNonLoraWeights { get; set; }

        public DateTime SavedAtUtc { get; set; } = DateTime.UtcNow;

        public ushort FormatVersion { get; set; }

        public bool Compressed { get; set; }
    }
}
