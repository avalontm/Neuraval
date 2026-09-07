using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

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

        // $7E:0019 crudo (0=chico, 1=grande, 2=capa, 3=fuego). Ver
        // MarioAgent.cs para como se codifica como entrada de la red, y
        // mario_bridge.lua para de donde sale.
        public int PowerupLevel { get; }

        // Indice del nivel/savestate activo (ver SAVESTATE_FILES en
        // mario_bridge.lua). Por ahora es solo informativo/telemetria -- no
        // se lo pasamos a la red como entrada (ver comentario en
        // MarioAgent.cs sobre por que).
        public int LevelIndex { get; }

        public IReadOnlyList<bool> Tiles { get; }
        public IReadOnlyList<SnesSprite> Sprites { get; }

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
            IReadOnlyList<bool> tiles,
            IReadOnlyList<SnesSprite> sprites)
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
        }

        public static SnesState Parse(string raw)
        {
            var parts = raw.Split('|');

            if (parts.Length != 14)
            {
                throw new FormatException(
                    $"Se esperaban 14 campos separados por '|' pero se recibieron {parts.Length}. " +
                    $"Raw recibido ({raw.Length} chars): \"{raw}\". " +
                    "Verifica que el script Lua activo en BizHawk sea mario_bridge.lua (version con powerup " +
                    "y levelIndex) y que no haya tirado un error.");
            }

            var frame = int.Parse(parts[0], CultureInfo.InvariantCulture);
            var marioX = int.Parse(parts[1], CultureInfo.InvariantCulture);
            var marioY = int.Parse(parts[2], CultureInfo.InvariantCulture);
            var marioVelocityX = int.Parse(parts[3], CultureInfo.InvariantCulture);
            var marioVelocityY = int.Parse(parts[4], CultureInfo.InvariantCulture);
            var isDead = parts[5] == "1";
            var lives = int.Parse(parts[6], CultureInfo.InvariantCulture);
            var tiles = parts[7].Select(character => character == '1').ToArray();
            var sprites = ParseSprites(parts[8]);
            var isGrounded = parts[9] == "1";
            var isLevelComplete = parts[10] == "1";
            var manualResetRequested = parts[11] == "1";
            var powerupLevel = int.Parse(parts[12], CultureInfo.InvariantCulture);
            var levelIndex = int.Parse(parts[13], CultureInfo.InvariantCulture);

            return new SnesState(
                frame,
                marioX,
                marioY,
                marioVelocityX,
                marioVelocityY,
                isDead,
                lives,
                isLevelComplete,
                manualResetRequested,
                isGrounded,
                powerupLevel,
                levelIndex,
                tiles,
                sprites);
        }

        private static IReadOnlyList<SnesSprite> ParseSprites(string raw)
        {
            if (string.IsNullOrEmpty(raw))
            {
                return Array.Empty<SnesSprite>();
            }

            return raw
                .Split(';')
                .Select(entry =>
                {
                    var coordinates = entry.Split(',');
                    var x = int.Parse(coordinates[0], CultureInfo.InvariantCulture);
                    var y = int.Parse(coordinates[1], CultureInfo.InvariantCulture);
                    var type = coordinates.Length > 2
                        ? int.Parse(coordinates[2], CultureInfo.InvariantCulture)
                        : 0;
                    return new SnesSprite(x, y, type);
                })
                .ToArray();
        }
    }
}
