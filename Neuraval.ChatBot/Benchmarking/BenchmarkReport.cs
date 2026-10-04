using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.ChatBot.Benchmarking
{
    public sealed class BenchmarkReport
    {
        public string SubjectName { get; }
        public IReadOnlyList<BenchmarkPromptResult> Results { get; }
        public long? ParameterCount { get; }
        public long? ModelSizeBytes { get; }
        public long? ProcessRamBytes { get; }
        public long? VramBytes { get; private set; }
        public double? Loss { get; private set; }
        public double? Perplexity { get; private set; }

        public double? TimeToFirstTokenSeconds => null;

        public int SuccessCount => Results.Count(r => r.Succeeded);
        public int FailureCount => Results.Count(r => !r.Succeeded);

        public double TotalElapsedSeconds => Results.Where(r => r.Succeeded).Sum(r => r.ElapsedSeconds);
        public long TotalCompletionTokens => Results.Where(r => r.Succeeded).Sum(r => (long)r.CompletionTokenCount);

        public double AverageLatencySeconds
        {
            get
            {
                var successful = Results.Where(r => r.Succeeded).ToList();
                return successful.Count > 0 ? successful.Average(r => r.ElapsedSeconds) : 0;
            }
        }

        public double TokensPerSecond => TotalElapsedSeconds > 0 ? TotalCompletionTokens / TotalElapsedSeconds : 0;

        public BenchmarkReport(
            string subjectName,
            IReadOnlyList<BenchmarkPromptResult> results,
            long? parameterCount,
            long? modelSizeBytes,
            long? processRamBytes)
        {
            if (string.IsNullOrWhiteSpace(subjectName))
                throw new ArgumentException("subjectName is required", nameof(subjectName));

            SubjectName = subjectName;
            Results = results ?? throw new ArgumentNullException(nameof(results));
            ParameterCount = parameterCount;
            ModelSizeBytes = modelSizeBytes;
            ProcessRamBytes = processRamBytes;
        }

        public BenchmarkReport WithVram(long vramBytes)
        {
            VramBytes = vramBytes;
            return this;
        }

        public BenchmarkReport WithEvaluationMetrics(double loss, double perplexity)
        {
            Loss = loss;
            Perplexity = perplexity;
            return this;
        }
    }
}
