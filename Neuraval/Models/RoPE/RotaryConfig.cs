using System;

namespace Neuraval.Core.Models.RoPE
{
    public sealed class RotaryConfig
    {
        public int HeadDim { get; set; }
        public int MaxPositionEmbeddings { get; set; } = 2048;
        public float RopeTheta { get; set; } = 1000000f;

        public void Validate()
        {
            if (HeadDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(HeadDim));

            if (HeadDim % 2 != 0)
                throw new InvalidOperationException($"HeadDim ({HeadDim}) debe ser par para aplicar RoPE");

            if (MaxPositionEmbeddings <= 0)
                throw new ArgumentOutOfRangeException(nameof(MaxPositionEmbeddings));

            if (RopeTheta <= 0f)
                throw new InvalidOperationException("RopeTheta debe ser positivo");
        }
    }
}
