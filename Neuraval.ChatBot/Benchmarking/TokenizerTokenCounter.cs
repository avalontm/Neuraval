using System;
using Neuraval.Core.Services;

namespace Neuraval.ChatBot.Benchmarking
{
    public sealed class TokenizerTokenCounter : ITokenCounter
    {
        private readonly ITokenizer _tokenizer;

        public TokenizerTokenCounter(ITokenizer tokenizer)
        {
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        }

        public int CountTokens(string text)
        {
            if (string.IsNullOrEmpty(text))
                return 0;

            return _tokenizer.Encode(text, addSpecialTokens: false).Length;
        }
    }
}
