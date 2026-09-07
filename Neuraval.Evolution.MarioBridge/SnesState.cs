using System.Globalization;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class SnesState
    {
        public const int GridRadius = 6;
        public const int GridSize = GridRadius * 2 + 1;

        public int Frame { get; }
        public int MarioX { get; }
        public int MarioY { get; }
        public int MarioVelocityX { get; }
        public int MarioVelocityY { get; }
        public bool IsDead { get; }
        public int Lives { get; }
        public bool IsLevelComplete { get; }
        public bool ManualResetRequested { get; }
        public bool IsGrounded { get; }
        public int PowerupLevel { get; }
        public int LevelIndex { get; }
        public int Direction { get; }
        public int Blocked { get; }
        public int SubPixelX { get; }
        public int SubPixelY { get; }
        public int AirState { get; }
        public int Ducking { get; }
        public int Climbing { get; }
        public int Water { get; }
        public int PMeter { get; }
        public int TakeoffMeter { get; }
        public int HurtTimer { get; }
        public int CapeTimer { get; }
        public int CameraX { get; }
        public int CameraY { get; }
        public int Layer2X { get; }
        public int Layer2Y { get; }
        public int Layer3X { get; }
        public int Layer3Y { get; }
        public int GameMode { get; }
        public int LevelMode { get; }
        public int Translevel { get; }
        public int Coins { get; }
        public int ItemBox { get; }
        public int Controller1 { get; }
        public int Controller1Prev { get; }
        public int Controller2 { get; }
        public int Controller2Prev { get; }
        public int MarioSubSpeed { get; }
        public int ReservedItemBox { get; }
        public int Controller1Copy { get; }
        public int Controller2Copy { get; }
        public int SpriteFrame { get; }
        public int Lag { get; }
        public int EmuFrame { get; }
        public int P2Controller1 { get; }
        public int P2Controller1Prev { get; }
        public int P2Controller2 { get; }
        public int P2Controller2Prev { get; }
        public int BluePowTimer { get; }
        public int SilverPowTimer { get; }
        public int DoorExitCounter { get; }
        public int ItemMemory { get; }
        public int CurrentPlayer { get; }
        public int Character { get; }
        public int CurrentPlayerCoins { get; }

        // $7E:1470 "Carrying something flag" y $7E:148F "Flag used to detect if Mario
        // holds an object" (ver SMW_RAM_Map_IA.md, seccion de items sostenibles).
        // Distinto de cero mientras Mario esta agarrando/cargando un sprite.
        public int CarryingFlag { get; }
        public int HoldingObjectFlag { get; }
        public bool IsHoldingItem => CarryingFlag != 0 || HoldingObjectFlag != 0;

        // $7E:13CE "Midway Point flag": true el frame en que Mario activo la barra
        // de punto medio (checkpoint) del nivel actual.
        public int MidwayPointFlag { get; }
        public bool MidwayPointReached => MidwayPointFlag != 0;

        // $7E:1420 "Yoshi Coins collected": contador exacto (0-5+) de monedas
        // Yoshi recogidas en el nivel actual. Ver SMW_RAM_Map_IA.md.
        public int YoshiCoinsCollected { get; }

        public int WallAheadDistance { get; }
        public int SolidAboveDistance { get; }
        public int SolidBelowDistance { get; }

        // $7E:1412 "Vertical scroll flag header": 0 = deshabilitado, 1 = habilitado,
        // 2 = habilitado condicionalmente (volando/trepando/etc). Se trata como
        // booleano: distinto de cero implica que el tramo actual admite scroll
        // vertical. Ver SMW_RAM_Map_IA.md.
        public bool IsVerticalLevel { get; }

        // $1C800/$1D800 tile IDs 0x0137/0x0138 (tiles superiores de tuberia
        // vertical exit-enabled, confirmado contra el tutorial oficial
        // "Introduction to Map16" de SMW Central; ver SMW_RAM_Map_IA.md).
        // Cuenta + dx/dy de la tuberia mas cercana que lleva a otra
        // seccion/piso al presionar Abajo sobre ella.
        public int PipeNear { get; }
        public int NearestPipeDx { get; }
        public int NearestPipeDy { get; }

        public IReadOnlyList<byte> Tiles { get; }
        public IReadOnlyList<SnesSprite> Sprites { get; }
        public IReadOnlyList<SnesSprite> ClusterSprites { get; }
        public int CoinsNear { get; }
        public int NearestCoinDx { get; }
        public int NearestCoinDy { get; }
        public int CoinBlocksNear { get; }
        public int NearestCoinBlockDx { get; }
        public int NearestCoinBlockDy { get; }
        public int DialogNear { get; }
        public int NearestDialogDx { get; }
        public int NearestDialogDy { get; }
        public IReadOnlyList<CliffGap> CliffGaps { get; }

        public SnesState(
            int frame,
            int marioX,
            int marioY,
            int marioVelocityX,
            int marioVelocityY,
            bool isDead,
            int lives,
            bool isLevelComplete,
            bool manualResetRequested,
            bool isGrounded,
            int powerupLevel,
            int levelIndex,
            IReadOnlyList<byte> tiles,
            IReadOnlyList<SnesSprite> sprites,
            int direction,
            int blocked,
            int subPixelX,
            int subPixelY,
            int airState,
            int ducking,
            int climbing,
            int water,
            int pMeter,
            int takeoffMeter,
            int hurtTimer,
            int capeTimer,
            int cameraX,
            int cameraY,
            int gameMode,
            int levelMode,
            int translevel,
            int coins,
            int itemBox,
            int controller1,
            int controller1Prev,
            int controller2,
            int controller2Prev,
            int marioSubSpeed = 0,
            int reservedItemBox = 0,
            int controller1Copy = 0,
            int controller2Copy = 0,
            int spriteFrame = 0,
            int lag = 0,
            int emuFrame = 0,
            IReadOnlyList<SnesSprite>? clusterSprites = null,
            int layer2X = 0,
            int layer2Y = 0,
            int layer3X = 0,
            int layer3Y = 0,
            int p2Controller1 = 0,
            int p2Controller1Prev = 0,
            int p2Controller2 = 0,
            int p2Controller2Prev = 0,
            int bluePowTimer = 0,
            int silverPowTimer = 0,
            int doorExitCounter = 0,
            int itemMemory = 0,
            int currentPlayer = 0,
            int character = 0,
            int currentPlayerCoins = 0,
            int coinsNear = 0,
            int nearestCoinDx = 0,
            int nearestCoinDy = 0,
            int coinBlocksNear = 0,
            int nearestCoinBlockDx = 0,
            int nearestCoinBlockDy = 0,
            int dialogNear = 0,
            int nearestDialogDx = 0,
            int nearestDialogDy = 0,
            IReadOnlyList<CliffGap>? cliffGaps = null,
            int carryingFlag = 0,
            int holdingObjectFlag = 0,
            int midwayPointFlag = 0,
            int yoshiCoinsCollected = 0,
            int wallAheadDistance = 0,
            int solidAboveDistance = 0,
            int solidBelowDistance = 0,
            bool isVerticalLevel = false,
            int pipeNear = 0,
            int nearestPipeDx = 0,
            int nearestPipeDy = 0)
        {
            Frame = frame;
            MarioX = marioX;
            MarioY = marioY;
            MarioVelocityX = marioVelocityX;
            MarioVelocityY = marioVelocityY;
            IsDead = isDead;
            Lives = lives;
            IsLevelComplete = isLevelComplete;
            ManualResetRequested = manualResetRequested;
            IsGrounded = isGrounded;
            PowerupLevel = powerupLevel;
            LevelIndex = levelIndex;
            Tiles = tiles;
            Sprites = sprites;
            Direction = direction;
            Blocked = blocked;
            SubPixelX = subPixelX;
            SubPixelY = subPixelY;
            AirState = airState;
            Ducking = ducking;
            Climbing = climbing;
            Water = water;
            PMeter = pMeter;
            TakeoffMeter = takeoffMeter;
            HurtTimer = hurtTimer;
            CapeTimer = capeTimer;
            CameraX = cameraX;
            CameraY = cameraY;
            GameMode = gameMode;
            LevelMode = levelMode;
            Translevel = translevel;
            Coins = coins;
            ItemBox = itemBox;
            Controller1 = controller1;
            Controller1Prev = controller1Prev;
            Controller2 = controller2;
            Controller2Prev = controller2Prev;
            MarioSubSpeed = marioSubSpeed;
            ReservedItemBox = reservedItemBox;
            Controller1Copy = controller1Copy;
            Controller2Copy = controller2Copy;
            SpriteFrame = spriteFrame;
            Lag = lag;
            EmuFrame = emuFrame;
            ClusterSprites = clusterSprites ?? Array.Empty<SnesSprite>();
            Layer2X = layer2X;
            Layer2Y = layer2Y;
            Layer3X = layer3X;
            Layer3Y = layer3Y;
            P2Controller1 = p2Controller1;
            P2Controller1Prev = p2Controller1Prev;
            P2Controller2 = p2Controller2;
            P2Controller2Prev = p2Controller2Prev;
            BluePowTimer = bluePowTimer;
            SilverPowTimer = silverPowTimer;
            DoorExitCounter = doorExitCounter;
            ItemMemory = itemMemory;
            CurrentPlayer = currentPlayer;
            Character = character;
            CurrentPlayerCoins = currentPlayerCoins;
            CoinsNear = coinsNear;
            NearestCoinDx = nearestCoinDx;
            NearestCoinDy = nearestCoinDy;
            CoinBlocksNear = coinBlocksNear;
            NearestCoinBlockDx = nearestCoinBlockDx;
            NearestCoinBlockDy = nearestCoinBlockDy;
            DialogNear = dialogNear;
            NearestDialogDx = nearestDialogDx;
            NearestDialogDy = nearestDialogDy;
            CliffGaps = cliffGaps ?? Array.Empty<CliffGap>();
            CarryingFlag = carryingFlag;
            HoldingObjectFlag = holdingObjectFlag;
            MidwayPointFlag = midwayPointFlag;
            YoshiCoinsCollected = yoshiCoinsCollected;
            WallAheadDistance = wallAheadDistance;
            SolidAboveDistance = solidAboveDistance;
            SolidBelowDistance = solidBelowDistance;
            IsVerticalLevel = isVerticalLevel;
            PipeNear = pipeNear;
            NearestPipeDx = nearestPipeDx;
            NearestPipeDy = nearestPipeDy;
        }

        public static SnesState Parse(string raw)
        {
            var parts = raw.Split('|');

            if (parts.Length != 70)
            {
                throw new FormatException(
                    $"Se esperaban 70 campos separados por '|' pero se recibieron {parts.Length}. " +
                    $"Raw recibido ({raw.Length} chars): \"{raw}\". " +
                    "Verifica que el script Lua activo en BizHawk sea mario_bridge.lua (protocolo v12 con " +
                    "state de juego, controles, sprites con status/stun/flags/misc/offscreen-full/eaten/" +
                    "object-interaction/spin, cluster sprites, capas, segundo jugador, timers " +
                    "de POW/door/player, senales de monedas/bloques-moneda, bloques de dialogo, " +
                    "acantilados, banderas de item sostenido / checkpoint (punto medio), contador de " +
                    "monedas Yoshi, senales de pared adelante / hueco-plataforma vertical arriba-abajo, " +
                    "bandera de nivel vertical y senal de tuberia vertical cercana) y que no haya tirado " +
                    "un error.");
            }

            return new SnesState(
                int.Parse(parts[0], CultureInfo.InvariantCulture),
                int.Parse(parts[1], CultureInfo.InvariantCulture),
                int.Parse(parts[2], CultureInfo.InvariantCulture),
                int.Parse(parts[3], CultureInfo.InvariantCulture),
                int.Parse(parts[4], CultureInfo.InvariantCulture),
                parts[5] == "1",
                int.Parse(parts[6], CultureInfo.InvariantCulture),
                parts[11] == "1",
                parts[12] == "1",
                parts[10] == "1",
                int.Parse(parts[13], CultureInfo.InvariantCulture),
                int.Parse(parts[14], CultureInfo.InvariantCulture),
                parts[7].Split(',').Select(token => byte.Parse(token, CultureInfo.InvariantCulture)).ToArray(),
                ParseSprites(parts[8]),
                int.Parse(parts[15], CultureInfo.InvariantCulture),
                int.Parse(parts[16], CultureInfo.InvariantCulture),
                int.Parse(parts[17], CultureInfo.InvariantCulture),
                int.Parse(parts[18], CultureInfo.InvariantCulture),
                int.Parse(parts[19], CultureInfo.InvariantCulture),
                int.Parse(parts[20], CultureInfo.InvariantCulture),
                int.Parse(parts[21], CultureInfo.InvariantCulture),
                int.Parse(parts[22], CultureInfo.InvariantCulture),
                int.Parse(parts[23], CultureInfo.InvariantCulture),
                int.Parse(parts[24], CultureInfo.InvariantCulture),
                int.Parse(parts[25], CultureInfo.InvariantCulture),
                int.Parse(parts[26], CultureInfo.InvariantCulture),
                int.Parse(parts[30], CultureInfo.InvariantCulture),
                int.Parse(parts[31], CultureInfo.InvariantCulture),
                int.Parse(parts[36], CultureInfo.InvariantCulture),
                int.Parse(parts[37], CultureInfo.InvariantCulture),
                int.Parse(parts[38], CultureInfo.InvariantCulture),
                int.Parse(parts[43], CultureInfo.InvariantCulture),
                int.Parse(parts[44], CultureInfo.InvariantCulture),
                int.Parse(parts[45], CultureInfo.InvariantCulture),
                int.Parse(parts[46], CultureInfo.InvariantCulture),
                int.Parse(parts[47], CultureInfo.InvariantCulture),
                int.Parse(parts[48], CultureInfo.InvariantCulture),
                int.Parse(parts[49], CultureInfo.InvariantCulture),
                int.Parse(parts[50], CultureInfo.InvariantCulture),
                int.Parse(parts[51], CultureInfo.InvariantCulture),
                int.Parse(parts[52], CultureInfo.InvariantCulture),
                int.Parse(parts[53], CultureInfo.InvariantCulture),
                int.Parse(parts[54], CultureInfo.InvariantCulture),
                int.Parse(parts[55], CultureInfo.InvariantCulture),
                ParseClusters(parts[9]),
                int.Parse(parts[32], CultureInfo.InvariantCulture),
                int.Parse(parts[33], CultureInfo.InvariantCulture),
                int.Parse(parts[34], CultureInfo.InvariantCulture),
                int.Parse(parts[35], CultureInfo.InvariantCulture),
                int.Parse(parts[56], CultureInfo.InvariantCulture),
                int.Parse(parts[57], CultureInfo.InvariantCulture),
                int.Parse(parts[58], CultureInfo.InvariantCulture),
                int.Parse(parts[59], CultureInfo.InvariantCulture),
                int.Parse(parts[27], CultureInfo.InvariantCulture),
                int.Parse(parts[28], CultureInfo.InvariantCulture),
                int.Parse(parts[29], CultureInfo.InvariantCulture),
                int.Parse(parts[41], CultureInfo.InvariantCulture),
                int.Parse(parts[39], CultureInfo.InvariantCulture),
                int.Parse(parts[40], CultureInfo.InvariantCulture),
                int.Parse(parts[42], CultureInfo.InvariantCulture),
                ParseCoinSignals(parts[60], 0),
                ParseCoinSignals(parts[60], 1),
                ParseCoinSignals(parts[60], 2),
                ParseCoinSignals(parts[60], 3),
                ParseCoinSignals(parts[60], 4),
                ParseCoinSignals(parts[60], 5),
                ParseDialogSignals(parts[61], 0),
                ParseDialogSignals(parts[61], 1),
                ParseDialogSignals(parts[61], 2),
                ParseCliffGaps(parts[62]),
                ParseInt(parts, 63),
                ParseInt(parts, 64),
                ParseInt(parts, 65),
                ParseInt(parts, 66),
                ParseWallSignals(parts[67], 0),
                ParseWallSignals(parts[67], 1),
                ParseWallSignals(parts[67], 2),
                parts[68] == "1",
                ParsePipeSignals(parts[69], 0),
                ParsePipeSignals(parts[69], 1),
                ParsePipeSignals(parts[69], 2));
        }

        private static int ParsePipeSignals(string raw, int index)
        {
            var values = raw.Split(';');
            return index < values.Length && int.TryParse(values[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : 0;
        }

        private static int ParseWallSignals(string raw, int index)
        {
            var values = raw.Split(';');
            return index < values.Length && int.TryParse(values[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : 0;
        }

        private static int ParseDialogSignals(string raw, int index)
        {
            var values = raw.Split(';');
            return index < values.Length && int.TryParse(values[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : 0;
        }

        private static IReadOnlyList<CliffGap> ParseCliffGaps(string raw)
        {
            if (string.IsNullOrEmpty(raw))
            {
                return Array.Empty<CliffGap>();
            }

            return raw
                .Split(';')
                .Chunk(2)
                .Where(chunk => chunk.Length == 2)
                .Select(chunk =>
                {
                    var start = int.TryParse(chunk[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var s) ? s : 0;
                    var width = int.TryParse(chunk[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var w) ? w : 0;
                    return new CliffGap(start, width);
                })
                .Where(gap => gap.WidthTiles > 0)
                .ToArray();
        }

        private static int ParseCoinSignals(string raw, int index)
        {
            var values = raw.Split(';');
            return index < values.Length && int.TryParse(values[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : 0;
        }

        private static IReadOnlyList<SnesSprite> ParseSprites(string raw)
        {
            if (string.IsNullOrEmpty(raw))
            {
                return Array.Empty<SnesSprite>();
            }

            return raw
                .Split(';')
                .Select(ParseSprite)
                .ToArray();
        }

        private static SnesSprite ParseSprite(string entry)
        {
            var values = entry.Split(',');
            return new SnesSprite(
                ParseInt(values, 0),
                ParseInt(values, 1),
                ParseInt(values, 2),
                ParseInt(values, 3),
                ParseInt(values, 4),
                ParseInt(values, 5),
                ParseInt(values, 6),
                ParseInt(values, 7),
                ParseInt(values, 8),
                ParseInt(values, 9),
                ParseInt(values, 10),
                ParseInt(values, 11),
                ParseInt(values, 12),
                ParseInt(values, 13),
                ParseInt(values, 14),
                ParseInt(values, 15),
                ParseInt(values, 16),
                ParseInt(values, 17),
                ParseInt(values, 18),
                ParseInt(values, 19));
        }

        private static IReadOnlyList<SnesSprite> ParseClusters(string raw)
        {
            if (string.IsNullOrEmpty(raw))
            {
                return Array.Empty<SnesSprite>();
            }

            return raw
                .Split(';')
                .Select(entry =>
                {
                    var values = entry.Split(',');
                    return new SnesSprite(ParseInt(values, 0), ParseInt(values, 1), 0);
                })
                .ToArray();
        }

        private static int ParseInt(string[] values, int index)
        {
            return index < values.Length && int.TryParse(values[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
                ? value
                : 0;
        }
    }
}