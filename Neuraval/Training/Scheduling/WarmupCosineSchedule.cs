using System;

namespace Neuraval.Core.Training.Scheduling
{
    public sealed class WarmupCosineSchedule : LearningRateSchedule
    {
        private readonly WarmupCosineScheduleOptions _options;

        public WarmupCosineSchedule(WarmupCosineScheduleOptions options)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
        }

        public WarmupCosineScheduleOptions Options => _options;

        public override float LearningRateAt(int globalStep)
        {
            if (globalStep < 0)
                throw new ArgumentOutOfRangeException(nameof(globalStep), "globalStep no puede ser negativo");

            if (_options.WarmupSteps > 0 && globalStep < _options.WarmupSteps)
            {
                return _options.PeakLearningRate * (globalStep / (float)_options.WarmupSteps);
            }

            if (globalStep >= _options.MaxSteps)
            {
                return _options.MinLearningRate;
            }

            int decaySteps = _options.MaxSteps - _options.WarmupSteps;
            int stepsIntoDecay = globalStep - _options.WarmupSteps;
            float progress = stepsIntoDecay / (float)decaySteps;
            float cosine = 0.5f * (1f + MathF.Cos(MathF.PI * progress));

            return _options.MinLearningRate + (_options.PeakLearningRate - _options.MinLearningRate) * cosine;
        }
    }
}
