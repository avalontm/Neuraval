namespace Neuraval.Evolution.MarioBridge
{
    public sealed class SnesEnvironment : IEnvironment<SnesState, SnesAction>
    {
        private readonly SnesBridgeConnection _connection;
        private int _previousMarioX;
        private bool _connected;

        // Que nivel/savestate pedir en el proximo Reset(). Program.cs lo
        // actualiza antes de evaluar cada generacion (ver rotacion de
        // niveles ahi); si nadie lo toca, queda en 0 y siempre se entrena
        // sobre el primer savestate configurado en Lua, igual que antes de
        // esta funcionalidad.
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
