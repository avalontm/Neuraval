using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Core.Services;

namespace Neuraval.Tests
{
    internal sealed class FakeTokenizer : ITokenizer
    {
        public int VocabSize => 100;
        public int PadToken => 0;
        public int UnknownToken => 1;
        public int StartToken => 2;
        public int EndToken => 3;
        public int SepToken => 4;

        public void BuildVocabulary(List<string> texts) { }
        public void BuildVocabulary(List<string> texts, int vocabSize) { }

        public int[] Encode(string text, bool addSpecialTokens = true)
        {
            var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            var ids = words.Select((_, i) => i + 10).ToArray();
            return addSpecialTokens ? new[] { StartToken }.Concat(ids).Append(EndToken).ToArray() : ids;
        }

        public string Decode(int[] tokenIds, bool skipSpecialTokens = true) => string.Join(' ', tokenIds);
        public int[] EncodeCausalSequence(string prompt, string response) => Encode(prompt).Concat(Encode(response)).ToArray();
        public int[] EncodePrompt(string prompt) => Encode(prompt);
        public int[] PadSequence(int[] sequence, int maxLength, bool padLeft = false) => sequence;
        public List<int[]> PadBatch(List<int[]> sequences, int? maxLength = null) => sequences;
        public void SaveToFile(string filepath) { }
    }
}
