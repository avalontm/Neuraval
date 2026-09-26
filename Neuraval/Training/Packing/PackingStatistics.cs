using System;
using System.Collections.Generic;

namespace Neuraval.Core.Training.Packing
{
    public sealed class PackingStatistics
    {
        public PackingStatistics(long tokensSeen, long paddingTokens, long usefulTokens, double packingEfficiency)
        {
            TokensSeen = tokensSeen;
            PaddingTokens = paddingTokens;
            UsefulTokens = usefulTokens;
            PackingEfficiency = packingEfficiency;
        }

        public long TokensSeen { get; }

        public long PaddingTokens { get; }

        public long UsefulTokens { get; }

        public double PackingEfficiency { get; }

        public static PackingStatistics From(IReadOnlyList<PackedSequence> packedSequences)
        {
            if (packedSequences == null)
                throw new ArgumentNullException(nameof(packedSequences));

            long tokensSeen = 0;
            long paddingTokens = 0;

            foreach (var packedSequence in packedSequences)
            {
                tokensSeen += packedSequence.Capacity;
                paddingTokens += packedSequence.PaddingTokenCount;
            }

            long usefulTokens = tokensSeen - paddingTokens;
            double efficiency = tokensSeen == 0 ? 0d : (double)usefulTokens / tokensSeen;

            return new PackingStatistics(tokensSeen, paddingTokens, usefulTokens, efficiency);
        }
    }
}
