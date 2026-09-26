using System;
using System.Collections.Generic;

namespace Neuraval.Core.Training.Stability
{
    public static class GradientClipper
    {
        public static float ComputeGlobalNorm(IReadOnlyList<float[]> gradientTensors)
        {
            if (gradientTensors == null)
                throw new ArgumentNullException(nameof(gradientTensors));

            double sumOfSquares = 0.0;

            foreach (var tensor in gradientTensors)
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    double value = tensor[i];
                    sumOfSquares += value * value;
                }
            }

            return (float)Math.Sqrt(sumOfSquares);
        }

        public static bool ContainsNaN(IReadOnlyList<float[]> gradientTensors)
        {
            if (gradientTensors == null)
                throw new ArgumentNullException(nameof(gradientTensors));

            foreach (var tensor in gradientTensors)
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    if (float.IsNaN(tensor[i]))
                        return true;
                }
            }

            return false;
        }

        public static bool ContainsInfinity(IReadOnlyList<float[]> gradientTensors)
        {
            if (gradientTensors == null)
                throw new ArgumentNullException(nameof(gradientTensors));

            foreach (var tensor in gradientTensors)
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    if (float.IsInfinity(tensor[i]))
                        return true;
                }
            }

            return false;
        }

        public static GradientStabilityReport ClipByGlobalNorm(IReadOnlyList<float[]> gradientTensors, GradientClipOptions options)
        {
            if (gradientTensors == null)
                throw new ArgumentNullException(nameof(gradientTensors));

            if (options == null)
                throw new ArgumentNullException(nameof(options));

            bool hasNaN = ContainsNaN(gradientTensors);
            bool hasInfinity = ContainsInfinity(gradientTensors);

            if (hasNaN || hasInfinity)
            {
                float unstableNorm = hasNaN ? float.NaN : float.PositiveInfinity;
                return GradientStabilityReport.Create(unstableNorm, unstableNorm, false, hasNaN, hasInfinity);
            }

            float normBeforeClip = ComputeGlobalNorm(gradientTensors);

            if (normBeforeClip <= options.MaxNorm || normBeforeClip == 0f)
            {
                return GradientStabilityReport.Create(normBeforeClip, normBeforeClip, false, false, false);
            }

            float scale = options.MaxNorm / normBeforeClip;

            foreach (var tensor in gradientTensors)
            {
                for (int i = 0; i < tensor.Length; i++)
                {
                    tensor[i] *= scale;
                }
            }

            return GradientStabilityReport.Create(normBeforeClip, options.MaxNorm, true, false, false);
        }
    }
}
