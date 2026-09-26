using System;
using Neuraval.Core.Services;

namespace Neuraval.ChatBot.Benchmarking
{
    /// <summary>
    /// Cuenta tokens usando el mismo <see cref="ITokenizer"/> que el modelo nativo,
    /// dando un conteo exacto en vez del aproximado de <see cref="WhitespaceTokenCounter"/>.
    /// </summary>
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
