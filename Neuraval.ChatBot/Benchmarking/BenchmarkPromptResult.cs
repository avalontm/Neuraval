namespace Neuraval.ChatBot.Benchmarking
{
    public sealed class BenchmarkPromptResult
    {
        public string Prompt { get; }
        public bool Succeeded { get; }
        public string? CompletionText { get; }
        public int PromptTokenCount { get; }
        public int CompletionTokenCount { get; }
        public double ElapsedSeconds { get; }
        public string? ErrorMessage { get; }

        private BenchmarkPromptResult(
            string prompt,
            bool succeeded,
            string? completionText,
            int promptTokenCount,
            int completionTokenCount,
            double elapsedSeconds,
            string? errorMessage)
        {
            Prompt = prompt;
            Succeeded = succeeded;
            CompletionText = completionText;
            PromptTokenCount = promptTokenCount;
            CompletionTokenCount = completionTokenCount;
            ElapsedSeconds = elapsedSeconds;
            ErrorMessage = errorMessage;
        }

        public static BenchmarkPromptResult Success(
            string prompt,
            string completionText,
            int promptTokenCount,
            int completionTokenCount,
            double elapsedSeconds)
        {
            return new BenchmarkPromptResult(prompt, true, completionText, promptTokenCount, completionTokenCount, elapsedSeconds, null);
        }

        public static BenchmarkPromptResult Failure(string prompt, double elapsedSeconds, string errorMessage)
        {
            return new BenchmarkPromptResult(prompt, false, null, 0, 0, elapsedSeconds, errorMessage);
        }
    }
}
