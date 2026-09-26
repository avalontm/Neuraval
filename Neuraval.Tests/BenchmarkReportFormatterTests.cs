using System.Collections.Generic;
using Neuraval.ChatBot.Benchmarking;
using Xunit;

namespace Neuraval.Tests
{
    public class BenchmarkReportFormatterTests
    {
        [Fact]
        public void FormatComparisonTable_IncludesEverySubjectName()
        {
            var reports = new List<BenchmarkReport>
            {
                new("Neuraval Native", new List<BenchmarkPromptResult>
                {
                    BenchmarkPromptResult.Success("p", "r", 1, 5, 1.0)
                }, parameterCount: 1_000_000, modelSizeBytes: 4_000_000, processRamBytes: 200_000_000),
                new("GPT/API", new List<BenchmarkPromptResult>
                {
                    BenchmarkPromptResult.Success("p", "r", 1, 8, 1.0)
                }, parameterCount: null, modelSizeBytes: null, processRamBytes: 200_000_000)
            };

            var table = BenchmarkReportFormatter.FormatComparisonTable(reports);

            Assert.Contains("Neuraval Native", table);
            Assert.Contains("GPT/API", table);
            Assert.Contains("n/d", table);
        }
    }
}
