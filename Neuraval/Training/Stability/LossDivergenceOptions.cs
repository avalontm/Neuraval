using System;

namespace Neuraval.Core.Training.Stability
{
    public sealed class LossDivergenceOptions
    {
        private LossDivergenceOptions(float divergenceFactor, int warmupSteps)
        {
            DivergenceFactor = divergenceFactor;
            WarmupSteps = warmupSteps;
        }

        public float DivergenceFactor { get; }

        public int WarmupSteps { get; }

        public static LossDivergenceOptions Create(float divergenceFactor = 2.0f, int warmupSteps = 10)
        {
            if (divergenceFactor <= 1f)
                throw new ArgumentOutOfRangeException(nameof(divergenceFactor), "divergenceFactor debe ser mayor que 1");

            if (warmupSteps < 0)
                throw new ArgumentOutOfRangeException(nameof(warmupSteps), "warmupSteps no puede ser negativo");

            return new LossDivergenceOptions(divergenceFactor, warmupSteps);
        }
    }
}
