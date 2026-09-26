using System;

namespace Neuraval.Core.Training.Scheduling
{
    public abstract class LearningRateSchedule
    {
        public abstract float LearningRateAt(int globalStep);

        public Func<int, float> AsFunc()
        {
            return LearningRateAt;
        }
    }
}
