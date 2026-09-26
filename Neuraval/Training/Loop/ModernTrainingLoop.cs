using System;
using System.Collections.Generic;

namespace Neuraval.Core.Training.Loop
{
    public sealed class ModernTrainingLoop
    {
        private readonly Func<int, TrainingStepOutcome> _executeStep;
        private readonly Func<int, float> _learningRateSchedule;
        private readonly Action<TrainingStepMetrics>? _onStepCompleted;
        private readonly Func<long?>? _gpuMemorySampler;
        private readonly List<TrainingStepMetrics> _history = new List<TrainingStepMetrics>();

        public ModernTrainingLoop(
            Func<int, TrainingStepOutcome> executeStep,
            Func<int, float> learningRateSchedule,
            Action<TrainingStepMetrics>? onStepCompleted = null,
            Func<long?>? gpuMemorySampler = null)
        {
            _executeStep = executeStep ?? throw new ArgumentNullException(nameof(executeStep));
            _learningRateSchedule = learningRateSchedule ?? throw new ArgumentNullException(nameof(learningRateSchedule));
            _onStepCompleted = onStepCompleted;
            _gpuMemorySampler = gpuMemorySampler;
        }

        public int GlobalStep { get; private set; }

        public long TokensSeen { get; private set; }

        public IReadOnlyList<TrainingStepMetrics> History => _history;

        public TrainingStepMetrics RunStep()
        {
            var outcome = _executeStep(GlobalStep);

            if (outcome == null)
                throw new InvalidOperationException("executeStep no puede devolver null");

            GlobalStep++;
            TokensSeen += outcome.TokensProcessed;

            float learningRate = _learningRateSchedule(GlobalStep);
            float tokensPerSecond = (float)(outcome.TokensProcessed / outcome.ElapsedSeconds);
            float samplesPerSecond = (float)(outcome.SamplesProcessed / outcome.ElapsedSeconds);
            long? gpuMemoryBytes = _gpuMemorySampler?.Invoke();

            var metrics = TrainingStepMetrics.Create(
                GlobalStep,
                TokensSeen,
                outcome.Loss,
                learningRate,
                outcome.GradientNorm,
                tokensPerSecond,
                samplesPerSecond,
                gpuMemoryBytes);

            _history.Add(metrics);
            _onStepCompleted?.Invoke(metrics);

            return metrics;
        }

        public IReadOnlyList<TrainingStepMetrics> Run(int steps)
        {
            if (steps < 0)
                throw new ArgumentOutOfRangeException(nameof(steps), "steps no puede ser negativo");

            var results = new List<TrainingStepMetrics>(steps);

            for (int i = 0; i < steps; i++)
            {
                results.Add(RunStep());
            }

            return results;
        }
    }
}
