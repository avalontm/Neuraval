using System;

namespace Neuraval.Core.Generation
{
    public sealed class CachedGenerationOutput
    {
        public GenerationResult Result { get; }
        public GenerationPerformanceReport Performance { get; }

        private CachedGenerationOutput(GenerationResult result, GenerationPerformanceReport performance)
        {
            Result = result;
            Performance = performance;
        }

        public static CachedGenerationOutput Create(GenerationResult result, GenerationPerformanceReport performance)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (performance == null)
                throw new ArgumentNullException(nameof(performance));

            return new CachedGenerationOutput(result, performance);
        }
    }
}
