using System;

namespace Neuraval.Core.Training.Scheduling
{
    public sealed class ConstantLearningRateSchedule : LearningRateSchedule
    {
        private readonly float _learningRate;

        public ConstantLearningRateSchedule(float learningRate)
        {
            if (learningRate < 0f)
                throw new ArgumentOutOfRangeException(nameof(learningRate), "learningRate no puede ser negativo");

            _learningRate = learningRate;
        }

        public override float LearningRateAt(int globalStep)
        {
            if (globalStep < 0)
                throw new ArgumentOutOfRangeException(nameof(globalStep), "globalStep no puede ser negativo");

            return _learningRate;
        }
    }
}
