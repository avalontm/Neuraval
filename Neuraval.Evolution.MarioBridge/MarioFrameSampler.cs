namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioFrameSampler
    {
        private SnesButton _lastAction;
        private int _lastX;
        private int _lastY;
        private int _lastPowerup;
        private bool _hasLast;

        public void Reset()
        {
            _hasLast = false;
        }

        public bool ShouldRecord(SnesState state, SnesButton action, bool terminal)
        {
            if (terminal)
            {
                return true;
            }

            var sameFingerprint = _hasLast
                && action == _lastAction
                && state.MarioX == _lastX
                && state.MarioY == _lastY
                && state.PowerupLevel == _lastPowerup;

            _hasLast = true;
            _lastAction = action;
            _lastX = state.MarioX;
            _lastY = state.MarioY;
            _lastPowerup = state.PowerupLevel;

            return !sameFingerprint;
        }
    }
}