using System;

namespace Neuraval.Core.Training.Loop
{
    public sealed class TrainingStepOutcome
    {
        public TrainingStepOutcome(float loss, float gradientNorm, int tokensProcessed, int samplesProcessed, double elapsedSeconds)
        {
            if (loss < 0f)
                throw new ArgumentOutOfRangeException(nameof(loss), "loss no puede ser negativo");

            if (tokensProcessed <= 0)
                throw new ArgumentOutOfRangeException(nameof(tokensProcessed), "tokensProcessed debe ser mayor que 0");

            if (samplesProcessed <= 0)
                throw new ArgumentOutOfRangeException(nameof(samplesProcessed), "samplesProcessed debe ser mayor que 0");

            if (elapsedSeconds <= 0d)
                throw new ArgumentOutOfRangeException(nameof(elapsedSeconds), "elapsedSeconds debe ser mayor que 0");

            Loss = loss;
            GradientNorm = gradientNorm;
            TokensProcessed = tokensProcessed;
            SamplesProcessed = samplesProcessed;
            ElapsedSeconds = elapsedSeconds;
        }

        public float Loss { get; }

        public float GradientNorm { get; }

        public int TokensProcessed { get; }

        public int SamplesProcessed { get; }

        public double ElapsedSeconds { get; }
    }
}
