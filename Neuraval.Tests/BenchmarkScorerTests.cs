using System.Collections.Generic;
using Neuraval.ChatBot.Benchmarking;
using Xunit;

namespace Neuraval.Tests
{
    public class BenchmarkScorerTests
    {
        private static BenchmarkReport MakeReport(string name, double completionTokens, double elapsedSeconds)
        {
            var results = new List<BenchmarkPromptResult>
            {
                BenchmarkPromptResult.Success("p", "r", 1, (int)completionTokens, elapsedSeconds)
            };
            return new BenchmarkReport(name, results, null, null, null);
        }

        [Fact]
        public void ComputeRelativeScores_GivesTheFastestSubjectTheTopThroughputScore()
        {
            var reports = new List<BenchmarkReport>
            {
                MakeReport("Neuraval Native", completionTokens: 100, elapsedSeconds: 1.0),
                MakeReport("GPT/API", completionTokens: 50, elapsedSeconds: 1.0)
            };

            var scores = BenchmarkScorer.ComputeRelativeScores(reports);

            Assert.Equal(100.0, scores[0].ThroughputScore);
            Assert.Equal(50.0, scores[1].ThroughputScore);
        }

        [Fact]
        public void ComputeRelativeScores_GivesTheLowestLatencySubjectTheTopLatencyScore()
        {
            var reports = new List<BenchmarkReport>
            {
                MakeReport("Neuraval Native", completionTokens: 10, elapsedSeconds: 1.0),
                MakeReport("Llama", completionTokens: 10, elapsedSeconds: 2.0)
            };

            var scores = BenchmarkScorer.ComputeRelativeScores(reports);

            Assert.Equal(100.0, scores[0].LatencyScore);
            Assert.Equal(50.0, scores[1].LatencyScore);
        }

        [Fact]
        public void ComputeRelativeScores_Throws_WhenReportsIsEmpty()
        {
            Assert.Throws<System.ArgumentException>(
                () => BenchmarkScorer.ComputeRelativeScores(new List<BenchmarkReport>()));
        }
    }
}
