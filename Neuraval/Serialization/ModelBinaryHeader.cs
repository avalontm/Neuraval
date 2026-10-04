using System;

namespace Neuraval.Core.Serialization
{
    public class ModelBinaryHeader
    {
        public int VocabSize { get; set; }

        public int EmbeddingDim { get; set; }

        public int NumLayers { get; set; }

        public int NumHeads { get; set; }

        public int FeedforwardDim { get; set; }

        public int MaxSequenceLength { get; set; }

        public int NumThreads { get; set; }

        public string TokenizerType { get; set; } = "wordlevel";

        public bool IsTrained { get; set; }

        public DateTime SavedAtUtc { get; set; } = DateTime.UtcNow;

        public ushort FormatVersion { get; set; }

        public bool Compressed { get; set; }

        public bool Quantized { get; set; }

        public string? QuantizationScheme { get; set; }
    }
}
