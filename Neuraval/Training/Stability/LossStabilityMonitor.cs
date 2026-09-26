namespace Neuraval.Core.Training.Stability
{
    public sealed class LossStabilityMonitor
    {
        private readonly LossDivergenceOptions _options;
        private float _bestLossSoFar = float.PositiveInfinity;
        private int _stepsObserved;

        public LossStabilityMonitor(LossDivergenceOptions? options = null)
        {
            _options = options ?? LossDivergenceOptions.Create();
        }

        public float BestLossSoFar => _bestLossSoFar;

        public int StepsObserved => _stepsObserved;

        public LossStabilityReport Observe(float loss)
        {
            _stepsObserved++;

            bool hasNaN = float.IsNaN(loss);
            bool hasInfinity = float.IsInfinity(loss);

            if (hasNaN || hasInfinity)
            {
                return LossStabilityReport.Create(loss, _bestLossSoFar, _stepsObserved, hasNaN, hasInfinity, false);
            }

            bool isDiverging = _stepsObserved > _options.WarmupSteps
                && !float.IsPositiveInfinity(_bestLossSoFar)
                && loss > _bestLossSoFar * _options.DivergenceFactor;

            if (loss < _bestLossSoFar)
            {
                _bestLossSoFar = loss;
            }

            return LossStabilityReport.Create(loss, _bestLossSoFar, _stepsObserved, false, false, isDiverging);
        }

        public void Reset()
        {
            _bestLossSoFar = float.PositiveInfinity;
            _stepsObserved = 0;
        }
    }
}
