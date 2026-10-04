using System.Collections.Generic;
using System.Text;

namespace Neuraval.Core.Tokenizers
{
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
