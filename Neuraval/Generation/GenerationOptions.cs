using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Core.Serialization.ModelExport;

namespace Neuraval.Core.Generation
{
    public sealed class GenerationOptions
    {
        public bool Greedy { get; }

        public float Temperature { get; }

        public float TopP { get; }

        public int TopK { get; }

        public float RepetitionPenalty { get; }

        public int MaxNewTokens { get; }

        public IReadOnlyList<int> StopTokenIds { get; }

        public int? Seed { get; }

        private GenerationOptions(
            bool greedy,
            float temperature,
            float topP,
            int topK,
            float repetitionPenalty,
            int maxNewTokens,
            IReadOnlyList<int> stopTokenIds,
            int? seed)
        {
            Greedy = greedy;
            Temperature = temperature;
            TopP = topP;
            TopK = topK;
            RepetitionPenalty = repetitionPenalty;
            MaxNewTokens = maxNewTokens;
            StopTokenIds = stopTokenIds;
            Seed = seed;
        }

        public static GenerationOptions Create(
            bool greedy,
            float temperature,
            float topP,
            int topK,
            float repetitionPenalty,
            int maxNewTokens,
            IReadOnlyList<int> stopTokenIds,
            int? seed = null)
        {
            if (temperature <= 0f)
                throw new ArgumentOutOfRangeException(nameof(temperature));

            if (topP <= 0f || topP > 1f)
                throw new ArgumentOutOfRangeException(nameof(topP));

            if (topK <= 0)
                throw new ArgumentOutOfRangeException(nameof(topK));

            if (repetitionPenalty <= 0f)
                throw new ArgumentOutOfRangeException(nameof(repetitionPenalty));

            if (maxNewTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxNewTokens));

            if (stopTokenIds == null)
                throw new ArgumentNullException(nameof(stopTokenIds));

            return new GenerationOptions(greedy, temperature, topP, topK, repetitionPenalty, maxNewTokens, stopTokenIds.ToArray(), seed);
        }

        public static GenerationOptions CreateGreedy(int maxNewTokens, IReadOnlyList<int> stopTokenIds, int? seed = null)
        {
            return Create(true, 1f, 1f, int.MaxValue, 1f, maxNewTokens, stopTokenIds, seed);
        }

        public static GenerationOptions Default(IReadOnlyList<int> stopTokenIds, int? seed = null)
        {
            return Create(false, 0.7f, 0.9f, 40, 1.1f, 256, stopTokenIds, seed);
        }

        public static GenerationOptions FromExportConfig(
            GenerationConfigOptions exportConfig,
            float repetitionPenalty,
            IReadOnlyList<int> stopTokenIds,
            bool greedy = false,
            int? seed = null)
        {
            if (exportConfig == null)
                throw new ArgumentNullException(nameof(exportConfig));

            return Create(greedy, exportConfig.Temperature, exportConfig.TopP, exportConfig.TopK, repetitionPenalty, exportConfig.MaxNewTokens, stopTokenIds, seed);
        }
    }
}
