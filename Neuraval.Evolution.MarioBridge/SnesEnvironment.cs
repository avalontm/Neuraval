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

        // Se llama despues de connection.Reconnect(), cuando el cliente de
        // Lua se reinicio y por lo tanto el proximo mensaje que llegue no es
        // la respuesta a nada que hayamos mandado nosotros, sino el primer
        // estado "de arranque" que Lua manda apenas conecta (igual que la
        // toda primera vez que se conecto BizHawk). Sin esto, el siguiente
        // Reset() intentaria mandar RESET antes de haber leido ese primer
        // estado, desincronizando el protocolo pregunta-respuesta.
        public void NotifyReconnected()
        {
            _connected = false;
        }

        // Caso inverso a NotifyReconnected(): se usa cuando este mismo
        // SnesEnvironment empieza a manejar una conexion que YA esta "en
        // medio" del protocolo pregunta-respuesta (por ejemplo, el modo
        // --learn reutiliza la conexion que veniamos usando para grabar al
        // humano jugar, en vez de abrir una nueva). En ese caso ya se
        // consumio el estado "de arranque" hace rato, asi que el primer
        // Reset() no debe volver a llamar ReceiveState() antes de mandar
        // RESET: BizHawk esta esperando NUESTRA respuesta al ultimo estado
        // que ya recibimos, no mandando uno nuevo por su cuenta. Sin esto,
        // Reset() se quedaria esperando datos que Lua nunca va a mandar
        // (deadlock).
        public void MarkConnected()
        {
            _connected = true;
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