using System;
using System.Collections.Generic;
using System.IO;

namespace Neuraval.Evolution.MarioBridge
{
    public static class MarioUnknownSpriteLogger
    {
        private static readonly string DefaultLogPath = Path.Combine("checkpoints", "unknown_sprites.log");

        public static string LogPath { get; set; } = DefaultLogPath;

        private static readonly HashSet<int> SeenThisSession = new();
        private static bool _loadedFromDisk;
        private static readonly object Lock = new();

        public static void Track(SnesState state)
        {
            if (state == null)
            {
                return;
            }

            TrackSprites(state.Sprites, state);
            TrackSprites(state.ClusterSprites, state);
        }

        private static void TrackSprites(IReadOnlyList<SnesSprite> sprites, SnesState state)
        {
            if (sprites == null)
            {
                return;
            }

            foreach (var sprite in sprites)
            {
                if (MarioSpriteNames.IsKnown(sprite.Type))
                {
                    continue;
                }

                TrackUnknown(sprite, state);
            }
        }

        private static void TrackUnknown(SnesSprite sprite, SnesState state)
        {
            lock (Lock)
            {
                EnsureLoadedFromDisk();

                if (!SeenThisSession.Add(sprite.Type))
                {
                    return;
                }

                var line = FormatLine(sprite, state);

                Console.WriteLine($"[MarioUnknownSpriteLogger] Sprite nuevo sin nombre: {line}");

                try
                {
                    var directory = Path.GetDirectoryName(LogPath);
                    if (!string.IsNullOrEmpty(directory))
                    {
                        Directory.CreateDirectory(directory);
                    }

                    File.AppendAllText(LogPath, line + Environment.NewLine);
                }
                catch (IOException)
                {
                }
            }
        }

        private static void EnsureLoadedFromDisk()
        {
            if (_loadedFromDisk)
            {
                return;
            }

            _loadedFromDisk = true;

            if (!File.Exists(LogPath))
            {
                return;
            }

            foreach (var existingLine in File.ReadLines(LogPath))
            {
                var id = ParseSpriteIdFromLine(existingLine);
                if (id.HasValue)
                {
                    SeenThisSession.Add(id.Value);
                }
            }
        }

        private static int? ParseSpriteIdFromLine(string line)
        {
            const string marker = "sprite=$";
            var index = line.IndexOf(marker, StringComparison.Ordinal);
            if (index < 0)
            {
                return null;
            }

            var start = index + marker.Length;
            var end = start;
            while (end < line.Length && Uri.IsHexDigit(line[end]))
            {
                end++;
            }

            if (end == start)
            {
                return null;
            }

            return int.TryParse(line.Substring(start, end - start), System.Globalization.NumberStyles.HexNumber, null, out var value)
                ? value
                : null;
        }

        private static string FormatLine(SnesSprite sprite, SnesState state)
        {
            return $"{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}Z sprite=${sprite.Type:X2} nivelIndex={state.LevelIndex} " +
                   $"marioX={state.MarioX} marioY={state.MarioY} spriteX={sprite.X} spriteY={sprite.Y} " +
                   $"status={sprite.Status} properties={sprite.Properties}";
        }

        public static void ResetSessionCache()
        {
            lock (Lock)
            {
                SeenThisSession.Clear();
                _loadedFromDisk = false;
            }
        }
    }
}
