using System;

namespace Neuraval.Core.Serialization.ModelExport
{
    public sealed class GenerationConfigOptions
    {
        public float Temperature { get; }

        public float TopP { get; }

        public int TopK { get; }

        public int MaxNewTokens { get; }

        private GenerationConfigOptions(float temperature, float topP, int topK, int maxNewTokens)
        {
            Temperature = temperature;
            TopP = topP;
            TopK = topK;
            MaxNewTokens = maxNewTokens;
        }

        public static GenerationConfigOptions Create(float temperature, float topP, int topK, int maxNewTokens)
        {
            if (temperature <= 0f)
                throw new ArgumentOutOfRangeException(nameof(temperature));

            if (topP <= 0f || topP > 1f)
                throw new ArgumentOutOfRangeException(nameof(topP));

            if (topK <= 0)
                throw new ArgumentOutOfRangeException(nameof(topK));

            if (maxNewTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxNewTokens));

            return new GenerationConfigOptions(temperature, topP, topK, maxNewTokens);
        }

        public static GenerationConfigOptions Default()
        {
            return new GenerationConfigOptions(0.7f, 0.9f, 40, 256);
        }
    }
}
