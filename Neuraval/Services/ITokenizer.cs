namespace Neuraval.Core.Services
{
    public interface ITokenizer
    {
        int VocabSize { get; }
        int PadToken { get; }
        int UnknownToken { get; }
        int StartToken { get; }
        int EndToken { get; }
        int SepToken { get; }

        void BuildVocabulary(List<string> texts);
        int[] Encode(string text, bool addSpecialTokens = true);
        string Decode(int[] tokenIds, bool skipSpecialTokens = true);
        int[] EncodeCausalSequence(string prompt, string response);
        int[] EncodePrompt(string prompt);
        int[] PadSequence(int[] sequence, int maxLength, bool padLeft = false);
        List<int[]> PadBatch(List<int[]> sequences, int? maxLength = null);
        void SaveToFile(string filepath);
    }
}
