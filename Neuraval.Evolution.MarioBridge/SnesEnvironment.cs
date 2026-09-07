namespace Neuraval.Evolution.MarioBridge
{
    public sealed class SnesEnvironment : IEnvironment<SnesState, SnesAction>
    {
        internal const float CoinReward = 50f;
        internal const float PowerupGainReward = 200f;
        internal const float PowerupLossPenalty = 150f;

        private readonly SnesBridgeConnection _connection;
        private int _previousMarioX;
        private int _previousCoins;
        private int _previousPowerup;
        private bool _connected;

        public int LevelIndex { get; set; }

        public SnesEnvironment(SnesBridgeConnection connection)
        {
            _connection = connection;
        }

        public SnesState Reset()
        {
            if (!_connected)
            {
                _connection.ReceiveState();
                _connected = true;
            }

            _connection.SendReset(LevelIndex);
            var state = _connection.ReceiveState();

            _previousMarioX = state.MarioX;
            _previousCoins = state.Coins;
            _previousPowerup = state.PowerupLevel;

            return state;
        }

        public EnvironmentStepResult<SnesState> Step(SnesAction action)
        {
            _connection.SendAction(action);
            var state = _connection.ReceiveState();

            var reward = (float)(state.MarioX - _previousMarioX);
            var coinGain = state.Coins - _previousCoins;
            var powerupDelta = state.PowerupLevel - _previousPowerup;

            if (coinGain > 0)
            {
                reward += coinGain * CoinReward;
            }

            if (powerupDelta > 0)
            {
                reward += powerupDelta * PowerupGainReward;
            }
            else if (powerupDelta < 0)
            {
                reward += powerupDelta * PowerupLossPenalty;
            }

            _previousMarioX = state.MarioX;
            _previousCoins = state.Coins;
            _previousPowerup = state.PowerupLevel;

            var done = state.IsDead || state.IsLevelComplete || state.ManualResetRequested;
            return new EnvironmentStepResult<SnesState>(state, reward, done);
        }
    }
}