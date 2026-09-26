using System.Collections.Generic;
using System.Text;

namespace Neuraval.Core.Tokenizers
{
    /// <summary>
    /// Divide un texto ya renderizado por un chat_template en segmentos de
    /// texto plano y "tokens especiales" reconocidos literalmente (p.ej.
    /// "&lt;s&gt;", "&lt;/s&gt;", "&lt;|im_start|&gt;").
    ///
    /// Esto hace falta porque un tokenizer BPE/SentencePiece normal NO
    /// reconstruye de forma fiable estos tokens a partir de su representación
    /// de texto: el algoritmo de merges/scores no está garantizado a
    /// recombinar "&lt;", "s", "&gt;" de vuelta en el token atómico "&lt;s&gt;", y aunque
    /// lo lograra, normalizaciones como el prefijo de espacio de SentencePiece
    /// romperían el resultado. Hay que reconocer estos tokens como unidades
    /// atómicas ANTES de tokenizar el resto del texto, igual que hacen
    /// llama.cpp/HuggingFace al tokenizar "con tokens especiales habilitados"
    /// (parse_special=true).
    /// </summary>
    public readonly struct SpecialTokenSegment
    {
        public bool IsSpecial { get; }
        public string Text { get; }
        public int TokenId { get; }

        public SpecialTokenSegment(bool isSpecial, string text, int tokenId)
        {
            IsSpecial = isSpecial;
            Text = text;
            TokenId = tokenId;
        }
    }

    public static class SpecialTokenTextSplitter
    {
        public static List<SpecialTokenSegment> Split(string text, IReadOnlyDictionary<string, int> specialTokens)
        {
            var result = new List<SpecialTokenSegment>();

            if (string.IsNullOrEmpty(text))
                return result;

            if (specialTokens.Count == 0)
            {
                result.Add(new SpecialTokenSegment(false, text, 0));
                return result;
            }

            // Se prueban primero los candidatos más largos para que, por ejemplo,
            // "<|im_end|>" no quede parcialmente capturado por un token más corto
            // que resulte ser prefijo suyo.
            var ordered = new List<string>(specialTokens.Keys);
            ordered.Sort((a, b) => b.Length.CompareTo(a.Length));

            var buffer = new StringBuilder();
            int i = 0;

            while (i < text.Length)
            {
                string? matched = null;

                foreach (var candidate in ordered)
                {
                    if (candidate.Length == 0)
                        continue;

                    if (i + candidate.Length > text.Length)
                        continue;

                    if (string.CompareOrdinal(text, i, candidate, 0, candidate.Length) == 0)
                    {
                        matched = candidate;
                        break;
                    }
                }

                if (matched != null)
                {
                    if (buffer.Length > 0)
                    {
                        result.Add(new SpecialTokenSegment(false, buffer.ToString(), 0));
                        buffer.Clear();
                    }

                    result.Add(new SpecialTokenSegment(true, matched, specialTokens[matched]));
                    i += matched.Length;
                }
                else
                {
                    buffer.Append(text[i]);
                    i++;
                }
            }

            if (buffer.Length > 0)
                result.Add(new SpecialTokenSegment(false, buffer.ToString(), 0));

            return result;
        }
    }
}
