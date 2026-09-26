using System;

namespace Neuraval.Core.Models.RoPE
{
    public sealed class RotaryEmbeddingCache
    {
        private readonly float[,] _cos;
        private readonly float[,] _sin;

        public int HeadDim { get; }
        public int HalfDim { get; }
        public int MaxPositionEmbeddings { get; }
        public float RopeTheta { get; }

        public RotaryEmbeddingCache(RotaryConfig config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            config.Validate();

            HeadDim = config.HeadDim;
            HalfDim = config.HeadDim / 2;
            MaxPositionEmbeddings = config.MaxPositionEmbeddings;
            RopeTheta = config.RopeTheta;

            _cos = new float[MaxPositionEmbeddings, HalfDim];
            _sin = new float[MaxPositionEmbeddings, HalfDim];

            for (int i = 0; i < HalfDim; i++)
            {
                float freq = 1f / MathF.Pow(RopeTheta, (2f * i) / HeadDim);

                for (int pos = 0; pos < MaxPositionEmbeddings; pos++)
                {
                    float angle = pos * freq;
                    _cos[pos, i] = MathF.Cos(angle);
                    _sin[pos, i] = MathF.Sin(angle);
                }
            }
        }

        public float CosAt(int position, int freqIndex)
        {
            EnsurePositionInRange(position);
            return _cos[position, freqIndex];
        }

        public float SinAt(int position, int freqIndex)
        {
            EnsurePositionInRange(position);
            return _sin[position, freqIndex];
        }

        private void EnsurePositionInRange(int position)
        {
            if (position < 0 || position >= MaxPositionEmbeddings)
                throw new ArgumentOutOfRangeException(
                    nameof(position),
                    $"La posición {position} excede MaxPositionEmbeddings ({MaxPositionEmbeddings})");
        }
    }
}
