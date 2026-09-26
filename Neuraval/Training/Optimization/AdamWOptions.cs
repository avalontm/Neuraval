using System;

namespace Neuraval.Core.Training.Optimization
{
    public sealed class AdamWOptions
    {
        private AdamWOptions(float learningRate, float weightDecay, float beta1, float beta2, float epsilon)
        {
            LearningRate = learningRate;
            WeightDecay = weightDecay;
            Beta1 = beta1;
            Beta2 = beta2;
            Epsilon = epsilon;
        }

        public float LearningRate { get; }

        public float WeightDecay { get; }

        public float Beta1 { get; }

        public float Beta2 { get; }

        public float Epsilon { get; }

        public static AdamWOptions Create(
            float learningRate = 0.0003f,
            float weightDecay = 0.1f,
            float beta1 = 0.9f,
            float beta2 = 0.95f,
            float epsilon = 1e-8f)
        {
            if (learningRate <= 0f)
                throw new ArgumentOutOfRangeException(nameof(learningRate), "learningRate debe ser mayor que cero");

            if (weightDecay < 0f)
                throw new ArgumentOutOfRangeException(nameof(weightDecay), "weightDecay no puede ser negativo");

            if (beta1 <= 0f || beta1 >= 1f)
                throw new ArgumentOutOfRangeException(nameof(beta1), "beta1 debe estar en (0, 1)");

            if (beta2 <= 0f || beta2 >= 1f)
                throw new ArgumentOutOfRangeException(nameof(beta2), "beta2 debe estar en (0, 1)");

            if (epsilon <= 0f)
                throw new ArgumentOutOfRangeException(nameof(epsilon), "epsilon debe ser mayor que cero");

            return new AdamWOptions(learningRate, weightDecay, beta1, beta2, epsilon);
        }
    }
}
