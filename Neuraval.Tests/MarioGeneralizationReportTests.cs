using System.Collections.Generic;
using System.Linq;
using Neuraval.Evolution.MarioBridge;
using Xunit;

namespace Neuraval.Tests
{
    // Cubre la agregacion pura (sin BizHawk) detras de --evaluate: el modo
    // que la Fase 5 del roadmap necesita para poder medir objetivamente si
    // un modelo "generaliza" (juega razonablemente bien en un nivel en el
    // que no entreno) en vez de solo mirar si completa DP1. RunEvaluateMode
    // solo llama RecordEpisode/PrintTo/ToCsvRows; toda la logica de conteo
    // vive aca para poder testearla sin abrir un socket.
    public class MarioGeneralizationReportTests
    {
        [Fact]
        public void RecordEpisode_AggregatesCompletionsAndBestX_PerLevel()
        {
            var report = new MarioGeneralizationReport();

            report.RecordEpisode(levelIndex: 0, bestX: 100, completed: false, MarioDeathCause.Enemy, stepsCapReached: false);
            report.RecordEpisode(levelIndex: 0, bestX: 300, completed: true, MarioDeathCause.None, stepsCapReached: false);
            report.RecordEpisode(levelIndex: 0, bestX: 200, completed: false, MarioDeathCause.FallOrHazard, stepsCapReached: false);

            var stats = report.ByLevel[0];

            Assert.Equal(3, stats.Episodes);
            Assert.Equal(1, stats.Completions);
            Assert.Equal(1, stats.DeathsByEnemy);
            Assert.Equal(1, stats.DeathsByFall);
            Assert.Equal(0, stats.StepsCapTerminations);
            Assert.Equal(300, stats.BestXMax);
            Assert.Equal(200f, stats.AverageBestX); // (100+300+200)/3 = 200
            Assert.True(System.Math.Abs(stats.CompletionPercent - 100f / 3f) < 0.01f);
        }

        [Fact]
        public void RecordEpisode_TracksMultipleLevelsIndependently()
        {
            // Este es el caso de uso central de la Fase 5: correr
            // --evaluate contra un nivel "conocido" (donde entreno) y uno
            // "held-out" (donde nunca entreno) en la misma corrida, y poder
            // comparar sus metricas por separado.
            var report = new MarioGeneralizationReport();

            report.RecordEpisode(levelIndex: 0, bestX: 4000, completed: true, MarioDeathCause.None, stepsCapReached: false);
            report.RecordEpisode(levelIndex: 1, bestX: 150, completed: false, MarioDeathCause.Enemy, stepsCapReached: false);
            report.RecordEpisode(levelIndex: 1, bestX: 180, completed: false, MarioDeathCause.Enemy, stepsCapReached: false);

            Assert.Equal(2, report.ByLevel.Count);

            var trained = report.ByLevel[0];
            Assert.Equal(1, trained.Episodes);
            Assert.Equal(100f, trained.CompletionPercent);

            var heldOut = report.ByLevel[1];
            Assert.Equal(2, heldOut.Episodes);
            Assert.Equal(0f, heldOut.CompletionPercent);
            Assert.Equal(2, heldOut.DeathsByEnemy);
            Assert.Equal(165f, heldOut.AverageBestX); // (150+180)/2
        }

        [Fact]
        public void RecordEpisode_CountsStepsCapOnlyWhenNotCompletedAndNotDead()
        {
            var report = new MarioGeneralizationReport();

            // Nunca deberia contarse steps-cap si el episodio de hecho
            // termino por muerte o por completar el nivel (RunEvaluateMode
            // ya garantiza esta exclusividad antes de llamar RecordEpisode,
            // pero el agregador la respeta igual si se la pasan mal).
            report.RecordEpisode(levelIndex: 0, bestX: 500, completed: true, MarioDeathCause.None, stepsCapReached: true);
            report.RecordEpisode(levelIndex: 0, bestX: 250, completed: false, MarioDeathCause.None, stepsCapReached: true);

            var stats = report.ByLevel[0];
            Assert.Equal(1, stats.Completions);
            Assert.Equal(1, stats.StepsCapTerminations);
        }

        [Fact]
        public void PrintTo_UsesProvidedLevelLabels_AndFallsBackToIndexWhenMissing()
        {
            var report = new MarioGeneralizationReport();
            report.RecordEpisode(levelIndex: 0, bestX: 100, completed: true, MarioDeathCause.None, stepsCapReached: false);
            report.RecordEpisode(levelIndex: 2, bestX: 50, completed: false, MarioDeathCause.Enemy, stepsCapReached: false);

            var lines = new List<string>();
            report.PrintTo(lines.Add, new[] { "DP1 (conocido)" });

            Assert.Contains(lines, line => line.StartsWith("DP1 (conocido)"));
            // El nivel 2 no tiene label en la lista (solo hay 1 elemento):
            // debe caer al indice como string, no tirar excepcion.
            Assert.Contains(lines, line => line.StartsWith("2 |"));
        }

        [Fact]
        public void ToCsvRows_OneRowPerLevel_WithQuotedModelAndLabel()
        {
            var report = new MarioGeneralizationReport();
            report.RecordEpisode(levelIndex: 0, bestX: 1200, completed: true, MarioDeathCause.None, stepsCapReached: false);
            report.RecordEpisode(levelIndex: 1, bestX: 300, completed: false, MarioDeathCause.FallOrHazard, stepsCapReached: false);

            var timestamp = new System.DateTime(2026, 9, 8, 12, 0, 0, System.DateTimeKind.Utc);
            var rows = report.ToCsvRows(
                timestamp,
                modelPath: "checkpoints/mario_best.navm",
                levelLabels: new[] { "Nivel, con coma", "Held-out" }).ToList();

            Assert.Equal(2, rows.Count);
            // El label del nivel 0 tiene una coma: debe ir citado entre
            // comillas para no romper el CSV.
            Assert.Contains("\"Nivel, con coma\"", rows[0]);
            Assert.Contains("Held-out", rows[1]);
            Assert.StartsWith("2026-09-08T12:00:00", rows[0]);
        }

        [Fact]
        public void ByLevel_IsEmpty_WhenNoEpisodesRecorded()
        {
            var report = new MarioGeneralizationReport();
            Assert.Empty(report.ByLevel);
        }
    }
}
