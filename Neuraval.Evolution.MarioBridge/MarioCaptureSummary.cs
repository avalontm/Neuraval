using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioCaptureSummary
    {
        private int _samples;
        private int _received;
        private int _episodes;
        private int _completed;
        private int _deaths;
        private int _enemyDeaths;
        private int _fallDeaths;
        private int _manualResets;
        private int _maxMarioX;
        private string _lastDeath = string.Empty;
        private readonly Dictionary<int, int> _episodesByLevel = new();
        private readonly Dictionary<int, int> _bestXByLevel = new();
        private readonly Dictionary<int, int> _yoshiByLevel = new();
        private readonly Dictionary<SnesButton, int> _actionCounts = new();

        public void RecordFrame(SnesState state, SnesButton action)
        {
            _received++;
            if (state.MarioX > _maxMarioX)
            {
                _maxMarioX = state.MarioX;
            }

            foreach (var button in Enum.GetValues<SnesButton>())
            {
                if (button != SnesButton.None && action.HasFlag(button))
                {
                    _actionCounts[button] = _actionCounts.GetValueOrDefault(button) + 1;
                }
            }
        }

        public void RecordSample()
        {
            _samples++;
        }

        public void RecordEpisode(SnesState terminalState, MarioTerminalReason reason)
        {
            var level = terminalState.LevelIndex;
            _episodes++;
            _episodesByLevel[level] = _episodesByLevel.GetValueOrDefault(level) + 1;

            if (!_bestXByLevel.ContainsKey(level) || terminalState.MarioX > _bestXByLevel[level])
            {
                _bestXByLevel[level] = terminalState.MarioX;
            }

            if (terminalState.YoshiCoinsCollected > _yoshiByLevel.GetValueOrDefault(level))
            {
                _yoshiByLevel[level] = terminalState.YoshiCoinsCollected;
            }

            switch (reason)
            {
                case MarioTerminalReason.LevelComplete:
                    _completed++;
                    break;
                case MarioTerminalReason.Death:
                    _deaths++;
                    if (MarioFitnessEvaluator.ClassifyDeathCause(terminalState) == MarioDeathCause.Enemy)
                    {
                        _enemyDeaths++;
                        _lastDeath = MarioFitnessEvaluator.DescribeDeath(terminalState);
                    }
                    else
                    {
                        _fallDeaths++;
                    }

                    break;
                case MarioTerminalReason.ManualReset:
                    _manualResets++;
                    break;
            }
        }

        public void Print()
        {
            Console.WriteLine("--- Resumen de captura ---");
            Console.WriteLine(
                $"{_received} frames jugados, {_samples} muestras grabadas (dedupe), {_episodes} episodios " +
                $"({_completed} completados, {_deaths} muertes [{_enemyDeaths} enemigo / {_fallDeaths} caida], {_manualResets} resets manuales), mejor MarioX {_maxMarioX}");

            if (!string.IsNullOrEmpty(_lastDeath))
            {
                Console.WriteLine($"Se murio mas a menudo contra: {_lastDeath}");
            }

            foreach (var entry in _bestXByLevel.OrderBy(e => e.Key))
            {
                var level = entry.Key;
                Console.WriteLine(
                    $"  Nivel {level}: {_episodesByLevel.GetValueOrDefault(level)} episodios, mejor MarioX {entry.Value}, maximas monedas Yoshi {_yoshiByLevel.GetValueOrDefault(level)}/5");
            }

            foreach (var entry in _actionCounts.OrderByDescending(e => e.Value))
            {
                var pct = 100.0 * entry.Value / Math.Max(1, _samples);
                Console.WriteLine($"  Boton {entry.Key}: {entry.Value} frames ({pct:F1}%)");
            }

            Console.WriteLine("--- Fin resumen ---");
        }
    }
}