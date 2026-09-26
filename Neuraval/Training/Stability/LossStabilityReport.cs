namespace Neuraval.Core.Training.Stability
{
    public sealed class LossStabilityReport
    {
        private LossStabilityReport(
            float loss,
            float bestLossSoFar,
            int stepsObserved,
            bool hasNaN,
            bool hasInfinity,
            bool isDiverging)
        {
            Loss = loss;
            BestLossSoFar = bestLossSoFar;
            StepsObserved = stepsObserved;
            HasNaN = hasNaN;
            HasInfinity = hasInfinity;
            IsDiverging = isDiverging;
        }

        public float Loss { get; }

        public float BestLossSoFar { get; }

        public int StepsObserved { get; }

        public bool HasNaN { get; }

        public bool HasInfinity { get; }

        public bool IsDiverging { get; }

        public bool IsStable => !HasNaN && !HasInfinity && !IsDiverging;

        public static LossStabilityReport Create(
            float loss,
            float bestLossSoFar,
            int stepsObserved,
            bool hasNaN,
            bool hasInfinity,
            bool isDiverging)
        {
            return new LossStabilityReport(loss, bestLossSoFar, stepsObserved, hasNaN, hasInfinity, isDiverging);
        }
    }
}
