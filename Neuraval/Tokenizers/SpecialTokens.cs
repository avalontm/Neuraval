namespace Neuraval.Core.Tokenizers
{
    public static class SpecialTokens
    {
        public const string Pad = "<|pad|>";
        public const string Unk = "<|unk|>";
        public const string Bos = "<|bos|>";
        public const string Eos = "<|eos|>";
        public const string Sep = "<|sep|>";
        public const string ImStart = "<|im_start|>";
        public const string ImEnd = "<|im_end|>";

        public static IReadOnlyList<string> All { get; } = new[] { Pad, Unk, Bos, Eos, Sep, ImStart, ImEnd };

        public static bool IsSpecialToken(string token)
        {
            return All.Contains(token);
        }
    }
}
