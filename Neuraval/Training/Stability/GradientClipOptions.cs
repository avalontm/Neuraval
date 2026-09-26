using System;

namespace Neuraval.Core.Training.Stability
{
    public sealed class GradientClipOptions
    {
        private GradientClipOptions(float maxNorm)
        {
            MaxNorm = maxNorm;
        }

        public float MaxNorm { get; }

        public static GradientClipOptions Create(float maxNorm)
        {
            if (maxNorm <= 0f)
                throw new ArgumentOutOfRangeException(nameof(maxNorm), "maxNorm debe ser mayor que cero");

            return new GradientClipOptions(maxNorm);
        }
    }
}
