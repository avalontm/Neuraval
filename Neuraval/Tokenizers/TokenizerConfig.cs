namespace Neuraval.Core.Tokenizers
{
    public sealed class TokenizerConfig
    {
        public int VocabSize { get; set; } = 32000;

        public int MinPairFrequency { get; set; } = 2;

        public void Validate()
        {
            if (VocabSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(VocabSize));

            if (MinPairFrequency <= 0)
                throw new ArgumentOutOfRangeException(nameof(MinPairFrequency));
        }
    }
}
