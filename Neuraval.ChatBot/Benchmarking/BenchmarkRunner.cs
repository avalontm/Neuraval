using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Neuraval.Abstractions;

namespace Neuraval.ChatBot.Benchmarking
{
    public static class BenchmarkRunner
    {
        public static async Task<BenchmarkReport> RunAsync(
            BenchmarkSubject subject,
            IReadOnlyList<string> prompts,
            CancellationToken cancellationToken = default)
        {
            if (subject is null)
                throw new ArgumentNullException(nameof(subject));

            if (prompts is null || prompts.Count == 0)
                throw new ArgumentException("prompts must contain at least one prompt", nameof(prompts));

            var results = new List<BenchmarkPromptResult>(prompts.Count);
            var stopwatch = new Stopwatch();

            foreach (var prompt in prompts)
            {
                var messages = new[] { new ChatMessage(ChatRole.User, prompt) };

                stopwatch.Restart();
                try
                {
                    var response = await subject.Model.SendAsync(messages, cancellationToken).ConfigureAwait(false);
                    stopwatch.Stop();

                    int promptTokens = subject.TokenCounter.CountTokens(prompt);
                    int completionTokens = subject.TokenCounter.CountTokens(response.Content);

                    results.Add(BenchmarkPromptResult.Success(
                        prompt,
                        response.Content,
                        promptTokens,
                        completionTokens,
                        stopwatch.Elapsed.TotalSeconds));
                }
                catch (Exception ex) when (!(ex is OperationCanceledException))
                {
                    stopwatch.Stop();
                    results.Add(BenchmarkPromptResult.Failure(prompt, stopwatch.Elapsed.TotalSeconds, ex.Message));
                }
            }

            long processRamBytes = Process.GetCurrentProcess().WorkingSet64;

            return new BenchmarkReport(
                subject.Name,
                results,
                subject.ParameterCount,
                subject.ModelSizeBytes,
                processRamBytes);
        }

        public static async Task<IReadOnlyList<BenchmarkReport>> RunAllAsync(
            IReadOnlyList<BenchmarkSubject> subjects,
            IReadOnlyList<string> prompts,
            CancellationToken cancellationToken = default)
        {
            if (subjects is null || subjects.Count == 0)
                throw new ArgumentException("subjects must contain at least one subject", nameof(subjects));

            var reports = new List<BenchmarkReport>(subjects.Count);
            foreach (var subject in subjects)
            {
                reports.Add(await RunAsync(subject, prompts, cancellationToken).ConfigureAwait(false));
            }

            return reports;
        }
    }
}
