using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace Neuraval.Evolution.MarioBridge
{
    // La Fase 5 del roadmap pide una prueba concreta de que el agente
    // "realmente aprendio a jugar" (y no memorizo DP1): que juegue
    // razonablemente bien en un nivel en el que no fue entrenado
    // especificamente. Esta clase agrega los resultados de episodios de
    // --evaluate (nivel, bestX, si completo, causa de muerte) sin ningun
    // I/O, para poder testear la logica de agregacion sin abrir un socket a
    // BizHawk (mismo criterio que MarioCurriculum en la Fase 3 y
    // MarioCaptureSummary): RunEvaluateMode solo la alimenta y la imprime.
    // No decide por si sola si un nivel es "conocido" o "held-out" — eso lo
    // sabe el usuario segun con que --level entreno, no algo que el codigo
    // pueda inferir.
    public sealed class MarioGeneralizationReport
    {
        public sealed class LevelStats
        {
            public int Episodes { get; internal set; }
            public int Completions { get; internal set; }
            public int DeathsByEnemy { get; internal set; }
            public int DeathsByFall { get; internal set; }
            public int StepsCapTerminations { get; internal set; }
            public int BestXMax { get; internal set; }
            public long BestXSum { get; internal set; }

            public float CompletionPercent => Episodes == 0 ? 0f : Completions * 100f / Episodes;
            public float AverageBestX => Episodes == 0 ? 0f : (float)BestXSum / Episodes;
        }

        private readonly Dictionary<int, LevelStats> _byLevel = new();

        public IReadOnlyDictionary<int, LevelStats> ByLevel => _byLevel;

        public void RecordEpisode(int levelIndex, int bestX, bool completed, MarioDeathCause deathCause, bool stepsCapReached)
        {
            if (!_byLevel.TryGetValue(levelIndex, out var stats))
            {
                stats = new LevelStats();
                _byLevel[levelIndex] = stats;
            }

            stats.Episodes++;
            stats.BestXSum += bestX;
            if (bestX > stats.BestXMax)
            {
                stats.BestXMax = bestX;
            }

            if (completed)
            {
                stats.Completions++;
            }
            else if (deathCause == MarioDeathCause.Enemy)
            {
                stats.DeathsByEnemy++;
            }
            else if (deathCause == MarioDeathCause.FallOrHazard)
            {
                stats.DeathsByFall++;
            }
            else if (stepsCapReached)
            {
                stats.StepsCapTerminations++;
            }
        }

        public void PrintTo(Action<string> writeLine, IReadOnlyList<string> levelLabels)
        {
            writeLine("Nivel | Episodios | % completado | Best X promedio | Best X maximo | Muertes enemigo | Muertes caida | Tope de pasos");

            foreach (var entry in _byLevel.OrderBy(e => e.Key))
            {
                var label = LabelFor(entry.Key, levelLabels);
                var stats = entry.Value;
                writeLine(
                    $"{label} | {stats.Episodes} | {stats.CompletionPercent:F1}% | {stats.AverageBestX:F1} | " +
                    $"{stats.BestXMax} | {stats.DeathsByEnemy} | {stats.DeathsByFall} | {stats.StepsCapTerminations}");
            }
        }

        public IEnumerable<string> ToCsvRows(DateTime timestampUtc, string modelPath, IReadOnlyList<string> levelLabels)
        {
            foreach (var entry in _byLevel.OrderBy(e => e.Key))
            {
                var label = LabelFor(entry.Key, levelLabels);
                var stats = entry.Value;
                yield return string.Join(
                    ",",
                    timestampUtc.ToString("O", CultureInfo.InvariantCulture),
                    CsvQuote(modelPath),
                    CsvQuote(label),
                    stats.Episodes,
                    stats.Completions,
                    stats.CompletionPercent.ToString("F1", CultureInfo.InvariantCulture),
                    stats.AverageBestX.ToString("F1", CultureInfo.InvariantCulture),
                    stats.BestXMax,
                    stats.DeathsByEnemy,
                    stats.DeathsByFall,
                    stats.StepsCapTerminations);
            }
        }

        private static string LabelFor(int levelIndex, IReadOnlyList<string> levelLabels)
        {
            return levelIndex < levelLabels.Count
                ? levelLabels[levelIndex]
                : levelIndex.ToString(CultureInfo.InvariantCulture);
        }

        private static string CsvQuote(string value)
        {
            if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n'))
            {
                return value;
            }

            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }
    }
}
