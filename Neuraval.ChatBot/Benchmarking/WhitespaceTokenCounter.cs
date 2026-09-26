using System;

namespace Neuraval.ChatBot.Benchmarking
{
    /// <summary>
    /// Aproxima el conteo de tokens partiendo el texto por espacios en blanco.
    /// No corresponde a la tokenización real de ningún modelo: se usa únicamente
    /// como fallback cuando el backend benchmarkeado no expone su propio tokenizer
    /// (por ejemplo, un endpoint OpenAI-compatible remoto). Los reportes que usan
    /// este contador deben marcarse como aproximados.
    /// </summary>
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
