using System;

namespace Neuraval.Core.Generation
{
    public sealed class GenerationPerformanceReport
    {
        public int PrefillTokenCount { get; }
        public double PrefillElapsedSeconds { get; }
        public int DecodeTokenCount { get; }
        public double DecodeElapsedSeconds { get; }
        public long CacheMemoryBytes { get; }

        public double PrefillTokensPerSecond =>
            PrefillElapsedSeconds > 0 ? PrefillTokenCount / PrefillElapsedSeconds : 0;

        public double DecodeTokensPerSecond =>
            DecodeElapsedSeconds > 0 ? DecodeTokenCount / DecodeElapsedSeconds : 0;

        private GenerationPerformanceReport(
            int prefillTokenCount,
            double prefillElapsedSeconds,
            int decodeTokenCount,
            double decodeElapsedSeconds,
            long cacheMemoryBytes)
        {
            PrefillTokenCount = prefillTokenCount;
            PrefillElapsedSeconds = prefillElapsedSeconds;
            DecodeTokenCount = decodeTokenCount;
            DecodeElapsedSeconds = decodeElapsedSeconds;
            CacheMemoryBytes = cacheMemoryBytes;
        }

        public static GenerationPerformanceReport Create(
            int prefillTokenCount,
            double prefillElapsedSeconds,
            int decodeTokenCount,
            double decodeElapsedSeconds,
            long cacheMemoryBytes)
        {
            if (prefillTokenCount < 0)
                throw new ArgumentOutOfRangeException(nameof(prefillTokenCount));

            if (decodeTokenCount < 0)
                throw new ArgumentOutOfRangeException(nameof(decodeTokenCount));

            if (prefillElapsedSeconds < 0)
                throw new ArgumentOutOfRangeException(nameof(prefillElapsedSeconds));

            if (decodeElapsedSeconds < 0)
                throw new ArgumentOutOfRangeException(nameof(decodeElapsedSeconds));

            if (cacheMemoryBytes < 0)
                throw new ArgumentOutOfRangeException(nameof(cacheMemoryBytes));

            return new GenerationPerformanceReport(
                prefillTokenCount,
                prefillElapsedSeconds,
                decodeTokenCount,
                decodeElapsedSeconds,
                cacheMemoryBytes);
        }
    }
}
