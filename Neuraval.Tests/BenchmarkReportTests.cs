using System.Collections.Generic;
using Neuraval.ChatBot.Benchmarking;
using Xunit;

namespace Neuraval.Tests
{
    public class BenchmarkReportTests
    {
        [Fact]
        public void TokensPerSecond_UsesOnlySuccessfulResults()
        {
            var results = new List<BenchmarkPromptResult>
            {
                BenchmarkPromptResult.Success("p1", "diez tokens de respuesta", promptTokenCount: 2, completionTokenCount: 10, elapsedSeconds: 2.0),
                BenchmarkPromptResult.Failure("p2", elapsedSeconds: 1.0, errorMessage: "timeout")
            };

            var report = new BenchmarkReport("Neuraval Native", results, parameterCount: null, modelSizeBytes: null, processRamBytes: null);

            Assert.Equal(5.0, report.TokensPerSecond);
            Assert.Equal(1, report.SuccessCount);
            Assert.Equal(1, report.FailureCount);
        }

        [Fact]
        public void TokensPerSecond_IsZero_WhenThereAreNoSuccessfulResults()
        {
            var results = new List<BenchmarkPromptResult>
            {
                BenchmarkPromptResult.Failure("p1", elapsedSeconds: 1.0, errorMessage: "error")
            };

            var report = new BenchmarkReport("GPT/API", results, null, null, null);

            Assert.Equal(0, report.TokensPerSecond);
            Assert.Equal(0, report.AverageLatencySeconds);
        }

        [Fact]
        public void AverageLatencySeconds_AveragesOnlySuccessfulPrompts()
        {
            var results = new List<BenchmarkPromptResult>
            {
                BenchmarkPromptResult.Success("p1", "ok", 1, 1, elapsedSeconds: 1.0),
                BenchmarkPromptResult.Success("p2", "ok", 1, 1, elapsedSeconds: 3.0),
                BenchmarkPromptResult.Failure("p3", elapsedSeconds: 100.0, errorMessage: "no cuenta")
            };

            var report = new BenchmarkReport("Llama", results, null, null, null);

            Assert.Equal(2.0, report.AverageLatencySeconds);
        }

        [Fact]
        public void TimeToFirstTokenSeconds_IsAlwaysNull_BecauseSendAsyncDoesNotStream()
        {
            var report = new BenchmarkReport("Gemma", new List<BenchmarkPromptResult>(), null, null, null);

            Assert.Null(report.TimeToFirstTokenSeconds);
        }

        [Fact]
        public void WithEvaluationMetrics_SetsLossAndPerplexity()
        {
            var report = new BenchmarkReport("Neuraval Native", new List<BenchmarkPromptResult>(), null, null, null);

            report.WithEvaluationMetrics(loss: 1.23, perplexity: 3.42);

            Assert.Equal(1.23, report.Loss);
            Assert.Equal(3.42, report.Perplexity);
        }

        [Fact]
        public void WithVram_SetsVramBytes()
        {
            var report = new BenchmarkReport("Qwen", new List<BenchmarkPromptResult>(), null, null, null);

            report.WithVram(1024);

            Assert.Equal(1024, report.VramBytes);
        }
    }
}
