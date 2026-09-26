using System;

namespace Neuraval.Core.Models.Architectures
{
    public enum ModelProfile
    {
        Nano,
        Small
    }

    public static class TransformerConfigProfiles
    {
        public const int DefaultVocabSize = 32000;
        public const int DefaultSeed = 42;

        public static TransformerConfig Create(ModelProfile profile, int vocabSize = DefaultVocabSize, int seed = DefaultSeed)
        {
            return profile switch
            {
                ModelProfile.Nano => Nano(vocabSize, seed),
                ModelProfile.Small => Small(vocabSize, seed),
                _ => throw new ArgumentOutOfRangeException(nameof(profile))
            };
        }

        public static TransformerConfig Nano(int vocabSize = DefaultVocabSize, int seed = DefaultSeed)
        {
            var config = new TransformerConfig
            {
                Architecture = "modern-decoder",
                VocabSize = vocabSize,
                HiddenSize = 256,
                NumHiddenLayers = 6,
                NumAttentionHeads = 8,
                NumKeyValueHeads = 2,
                IntermediateSize = 768,
                MaxPositionEmbeddings = 2048,
                RopeTheta = 10000f,
                RmsNormEps = 1e-6f,
                Activation = "silu",
                NormType = "rmsnorm",
                TieWordEmbeddings = true,
                Seed = seed
            };

            config.Validate();
            return config;
        }

        public static TransformerConfig Small(int vocabSize = DefaultVocabSize, int seed = DefaultSeed)
        {
            var config = new TransformerConfig
            {
                Architecture = "modern-decoder",
                VocabSize = vocabSize,
                HiddenSize = 512,
                NumHiddenLayers = 12,
                NumAttentionHeads = 8,
                NumKeyValueHeads = 2,
                IntermediateSize = 1408,
                MaxPositionEmbeddings = 4096,
                RopeTheta = 10000f,
                RmsNormEps = 1e-6f,
                Activation = "silu",
                NormType = "rmsnorm",
                TieWordEmbeddings = true,
                Seed = seed
            };

            config.Validate();
            return config;
        }
    }
}
