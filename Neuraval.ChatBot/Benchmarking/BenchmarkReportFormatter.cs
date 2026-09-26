using System.Collections.Generic;
using System.Linq;
using System.Text;
using Neuraval.Core.Utils;

namespace Neuraval.ChatBot.Benchmarking
{
    public static class BenchmarkReportFormatter
    {
        public static string FormatComparisonTable(IReadOnlyList<BenchmarkReport> reports)
        {
            var scores = BenchmarkScorer.ComputeRelativeScores(reports);
            var scoreByName = scores.ToDictionary(s => s.SubjectName, s => s.OverallScore);

            var sb = new StringBuilder();
            sb.AppendLine($"{"Backend",-20} {"tok/s",8} {"latencia(s)",12} {"params",12} {"tamaño",10} {"RAM",10} {"ok/fail",8} {"score",6}");
            sb.AppendLine(new string('-', 92));

            foreach (var report in reports)
            {
                string parameterCount = report.ParameterCount.HasValue ? report.ParameterCount.Value.ToString("N0") : "n/d";
                string modelSize = report.ModelSizeBytes.HasValue ? MemoryProfiler.FormatBytes(report.ModelSizeBytes.Value) : "n/d";
                string ram = report.ProcessRamBytes.HasValue ? MemoryProfiler.FormatBytes(report.ProcessRamBytes.Value) : "n/d";
                string okFail = $"{report.SuccessCount}/{report.FailureCount}";
                double score = scoreByName.TryGetValue(report.SubjectName, out var s) ? s : 0;

                sb.AppendLine(
                    $"{report.SubjectName,-20} {report.TokensPerSecond,8:F2} {report.AverageLatencySeconds,12:F3} {parameterCount,12} {modelSize,10} {ram,10} {okFail,8} {score,6:F1}");
            }

            sb.AppendLine();
            sb.AppendLine("Nota: time-to-first-token, VRAM, loss y perplexity no se muestran porque requieren");
            sb.AppendLine("streaming, telemetría de GPU y un dataset de evaluación (Fase 26) respectivamente.");

            return sb.ToString();
        }
    }
}
