using System;
using System.Text;

namespace Neuraval.Core.Training.Loop
{
    public sealed class TrainingStepMetrics
    {
        private TrainingStepMetrics(
            int globalStep,
            long tokensSeen,
            float loss,
            float perplexity,
            float learningRate,
            float gradientNorm,
            float tokensPerSecond,
            float samplesPerSecond,
            long? gpuMemoryBytes)
        {
            GlobalStep = globalStep;
            TokensSeen = tokensSeen;
            Loss = loss;
            Perplexity = perplexity;
            LearningRate = learningRate;
            GradientNorm = gradientNorm;
            TokensPerSecond = tokensPerSecond;
            SamplesPerSecond = samplesPerSecond;
            GpuMemoryBytes = gpuMemoryBytes;
        }

        public int GlobalStep { get; }

        public long TokensSeen { get; }

        public float Loss { get; }

        public float Perplexity { get; }

        public float LearningRate { get; }

        public float GradientNorm { get; }

        public float TokensPerSecond { get; }

        public float SamplesPerSecond { get; }

        public long? GpuMemoryBytes { get; }

        public static TrainingStepMetrics Create(
            int globalStep,
            long tokensSeen,
            float loss,
            float learningRate,
            float gradientNorm,
            float tokensPerSecond,
            float samplesPerSecond,
            long? gpuMemoryBytes = null)
        {
            if (globalStep < 0)
                throw new ArgumentOutOfRangeException(nameof(globalStep), "globalStep no puede ser negativo");

            if (tokensSeen < 0)
                throw new ArgumentOutOfRangeException(nameof(tokensSeen), "tokensSeen no puede ser negativo");

            if (loss < 0f)
                throw new ArgumentOutOfRangeException(nameof(loss), "loss no puede ser negativo");

            float perplexity = MathF.Exp(loss);

            return new TrainingStepMetrics(
                globalStep,
                tokensSeen,
                loss,
                perplexity,
                learningRate,
                gradientNorm,
                tokensPerSecond,
                samplesPerSecond,
                gpuMemoryBytes);
        }

        public string ToLogString()
        {
            var builder = new StringBuilder();
            builder.AppendLine($"Step: {GlobalStep}");
            builder.AppendLine($"Tokens: {TokensSeen:N0}");
            builder.AppendLine($"Loss: {Loss:F3}");
            builder.AppendLine($"Perplexity: {Perplexity:F2}");
            builder.AppendLine($"LR: {LearningRate:F6}");
            builder.AppendLine($"Grad norm: {GradientNorm:F2}");
            builder.Append($"Tokens/sec: {TokensPerSecond:N0}");

            return builder.ToString();
        }
    }
}
