using System.Collections.Generic;
using System.Threading.Tasks;
using Neuraval.ChatBot.Benchmarking;
using Xunit;

namespace Neuraval.Tests
{
    public class BenchmarkRunnerTests
    {
        [Fact]
        public async Task RunAsync_RecordsSuccessfulResult_WithTokenCountsAndElapsedTime()
        {
            var model = FakeChatModel.WithFixedReply("una dos tres");
            var subject = new BenchmarkSubject("Neuraval Native", model);

            var report = await BenchmarkRunner.RunAsync(subject, new List<string> { "hola mundo" });

            Assert.Equal(1, report.SuccessCount);
            Assert.Equal(0, report.FailureCount);
            Assert.Equal(3, report.Results[0].CompletionTokenCount);
            Assert.Equal(2, report.Results[0].PromptTokenCount);
            Assert.True(report.TotalElapsedSeconds >= 0);
            Assert.Equal("Neuraval Native", report.SubjectName);
        }

        [Fact]
        public async Task RunAsync_RecordsFailure_WhenModelThrowsChatModelException()
        {
            var model = FakeChatModel.ThatThrows("endpoint caído");
            var subject = new BenchmarkSubject("GPT/API", model);

            var report = await BenchmarkRunner.RunAsync(subject, new List<string> { "hola" });

            Assert.Equal(0, report.SuccessCount);
            Assert.Equal(1, report.FailureCount);
            Assert.False(report.Results[0].Succeeded);
            Assert.Equal("endpoint caído", report.Results[0].ErrorMessage);
        }

        [Fact]
        public async Task RunAsync_ContinuesWithRemainingPrompts_AfterOneFails()
        {
            var callCount = 0;
            var model = new FakeChatModel(() =>
            {
                callCount++;
                if (callCount == 1)
                    throw new Neuraval.Abstractions.ChatModelException("falla puntual");

                return new Neuraval.Abstractions.ChatMessage(Neuraval.Abstractions.ChatRole.Assistant, "ok");
            });

            var subject = new BenchmarkSubject("Llama", model);
            var report = await BenchmarkRunner.RunAsync(subject, new List<string> { "p1", "p2" });

            Assert.Equal(2, report.Results.Count);
            Assert.False(report.Results[0].Succeeded);
            Assert.True(report.Results[1].Succeeded);
        }

        [Fact]
        public async Task RunAsync_Throws_WhenPromptsIsEmpty()
        {
            var subject = new BenchmarkSubject("Gemma", FakeChatModel.WithFixedReply("hola"));

            await Assert.ThrowsAsync<System.ArgumentException>(
                () => BenchmarkRunner.RunAsync(subject, new List<string>()));
        }

        [Fact]
        public async Task RunAllAsync_RunsEverySubjectAgainstTheSamePrompts()
        {
            var subjects = new List<BenchmarkSubject>
            {
                new("Neuraval Native", FakeChatModel.WithFixedReply("a b")),
                new("Qwen", FakeChatModel.WithFixedReply("c d e"))
            };

            var reports = await BenchmarkRunner.RunAllAsync(subjects, new List<string> { "prompt único" });

            Assert.Equal(2, reports.Count);
            Assert.Equal("Neuraval Native", reports[0].SubjectName);
            Assert.Equal("Qwen", reports[1].SubjectName);
            Assert.Equal(2, reports[0].Results[0].CompletionTokenCount);
            Assert.Equal(3, reports[1].Results[0].CompletionTokenCount);
        }
    }
}
