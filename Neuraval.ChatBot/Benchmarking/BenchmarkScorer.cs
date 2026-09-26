using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.ChatBot.Benchmarking
{
    public sealed class BenchmarkScore
    {
        public string SubjectName { get; }
        public double ThroughputScore { get; }
        public double LatencyScore { get; }
        public double OverallScore { get; }

        public BenchmarkScore(string subjectName, double throughputScore, double latencyScore, double overallScore)
        {
            SubjectName = subjectName;
            ThroughputScore = throughputScore;
            LatencyScore = latencyScore;
            OverallScore = overallScore;
        }
    }

    /// <summary>
    /// Calcula un "benchmark score" (Fase 25) a partir de tokens/seg y latencia, normalizando cada
    /// reporte contra el mejor valor del grupo. No hay una fórmula universal de "score" para LLMs;
    /// esta es una combinación simple y explícita (promedio de throughput relativo y latencia relativa),
    /// pensada para comparar corridas del mismo set de prompts, no como métrica absoluta.
    /// </summary>
    public static class BenchmarkScorer
    {
        public static IReadOnlyList<BenchmarkScore> ComputeRelativeScores(IReadOnlyList<BenchmarkReport> reports)
        {
            if (reports is null || reports.Count == 0)
                throw new ArgumentException("reports must contain at least one report", nameof(reports));

            double bestTokensPerSecond = reports.Max(r => r.TokensPerSecond);
            double bestAverageLatency = reports
                .Where(r => r.AverageLatencySeconds > 0)
                .Select(r => r.AverageLatencySeconds)
                .DefaultIfEmpty(0)
                .Min();

            var scores = new List<BenchmarkScore>(reports.Count);
            foreach (var report in reports)
            {
                double throughputScore = bestTokensPerSecond > 0
                    ? 100.0 * report.TokensPerSecond / bestTokensPerSecond
                    : 0;

                double latencyScore = (bestAverageLatency > 0 && report.AverageLatencySeconds > 0)
                    ? 100.0 * bestAverageLatency / report.AverageLatencySeconds
                    : 0;

                double overallScore = (throughputScore + latencyScore) / 2.0;

                scores.Add(new BenchmarkScore(report.SubjectName, throughputScore, latencyScore, overallScore));
            }

            return scores;
        }
    }
}
