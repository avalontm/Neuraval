using System;
using System.Collections.Generic;

namespace Neuraval.Core.Models
{
    public class TransformerModelLoraState
    {
        public int NumLayers { get; set; }
        public int EmbeddingDim { get; set; }

        public bool FreezeNonLoraWeights { get; set; }

        public List<LoraAttentionState> BlockStates { get; set; } = new();
    }
}
