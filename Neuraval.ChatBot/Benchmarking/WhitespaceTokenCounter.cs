using System;

namespace Neuraval.ChatBot.Benchmarking
{
    public sealed class WhitespaceTokenCounter : ITokenCounter
    {
        public int CountTokens(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return 0;

            return text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;
        }
    }
}
