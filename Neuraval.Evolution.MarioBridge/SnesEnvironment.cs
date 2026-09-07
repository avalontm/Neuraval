namespace Neuraval.Evolution.MarioBridge
{
    public sealed class SnesEnvironment : IEnvironment<SnesState, SnesAction>
    {
        private readonly SnesBridgeConnection _connection;
        private int _previousMarioX;
        private bool _connected;

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

            _connection.SendReset();
            var state = _connection.ReceiveState();
            _previousMarioX = state.MarioX;
            return state;
        }

        public EnvironmentStepResult<SnesState> Step(SnesAction action)
        {
            _connection.SendAction(action);
            var state = _connection.ReceiveState();
            var reward = state.MarioX - _previousMarioX;
            _previousMarioX = state.MarioX;
            var done = state.IsDead || state.IsLevelComplete || state.ManualResetRequested;
            return new EnvironmentStepResult<SnesState>(state, reward, done);
        }
    }
}
