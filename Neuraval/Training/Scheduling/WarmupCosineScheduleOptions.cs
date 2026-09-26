using System;

namespace Neuraval.Core.Training.Scheduling
{
    public sealed class WarmupCosineScheduleOptions
    {
        private WarmupCosineScheduleOptions(float peakLearningRate, int warmupSteps, int maxSteps, float minLearningRate)
        {
            PeakLearningRate = peakLearningRate;
            WarmupSteps = warmupSteps;
            MaxSteps = maxSteps;
            MinLearningRate = minLearningRate;
        }

        public float PeakLearningRate { get; }

        public int WarmupSteps { get; }

        public int MaxSteps { get; }

        public float MinLearningRate { get; }

        public static WarmupCosineScheduleOptions Create(
            float peakLearningRate,
            int warmupSteps,
            int maxSteps,
            float minLearningRate)
        {
            if (peakLearningRate <= 0f)
                throw new ArgumentOutOfRangeException(nameof(peakLearningRate), "peakLearningRate debe ser mayor que cero");

            if (warmupSteps < 0)
                throw new ArgumentOutOfRangeException(nameof(warmupSteps), "warmupSteps no puede ser negativo");

            if (maxSteps <= warmupSteps)
                throw new ArgumentOutOfRangeException(nameof(maxSteps), "maxSteps debe ser mayor que warmupSteps");

            if (minLearningRate < 0f)
                throw new ArgumentOutOfRangeException(nameof(minLearningRate), "minLearningRate no puede ser negativo");

            if (minLearningRate > peakLearningRate)
                throw new ArgumentOutOfRangeException(nameof(minLearningRate), "minLearningRate no puede ser mayor que peakLearningRate");

            return new WarmupCosineScheduleOptions(peakLearningRate, warmupSteps, maxSteps, minLearningRate);
        }
    }
}
