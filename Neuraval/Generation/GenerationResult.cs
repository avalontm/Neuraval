using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Core.Generation
{
    public sealed class GenerationResult
    {
        public IReadOnlyList<int> GeneratedTokenIds { get; }

        public GenerationFinishReason FinishReason { get; }

        private GenerationResult(IReadOnlyList<int> generatedTokenIds, GenerationFinishReason finishReason)
        {
            GeneratedTokenIds = generatedTokenIds;
            FinishReason = finishReason;
        }

        public static GenerationResult Create(IReadOnlyList<int> generatedTokenIds, GenerationFinishReason finishReason)
        {
            if (generatedTokenIds == null)
                throw new ArgumentNullException(nameof(generatedTokenIds));

            return new GenerationResult(generatedTokenIds.ToArray(), finishReason);
        }
    }
}
