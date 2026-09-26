using System;
using Neuraval.Core.Generation;
using Xunit;

namespace Neuraval.Tests
{
    public class GenerationPerformanceReportTests
    {
        [Fact]
        public void Create_ValidValues_ComputesTokensPerSecond()
        {
            var report = GenerationPerformanceReport.Create(10, 2.0, 20, 4.0, 1024);

            Assert.Equal(5.0, report.PrefillTokensPerSecond, 3);
            Assert.Equal(5.0, report.DecodeTokensPerSecond, 3);
            Assert.Equal(1024, report.CacheMemoryBytes);
        }

        [Fact]
        public void Create_ZeroElapsedSeconds_ReturnsZeroTokensPerSecond()
        {
            var report = GenerationPerformanceReport.Create(10, 0, 0, 0, 512);

            Assert.Equal(0, report.PrefillTokensPerSecond);
            Assert.Equal(0, report.DecodeTokensPerSecond);
        }

        [Theory]
        [InlineData(-1, 1.0, 1, 1.0, 1)]
        [InlineData(1, -1.0, 1, 1.0, 1)]
        [InlineData(1, 1.0, -1, 1.0, 1)]
        [InlineData(1, 1.0, 1, -1.0, 1)]
        [InlineData(1, 1.0, 1, 1.0, -1)]
        public void Create_InvalidValues_Throws(int prefillTokens, double prefillSeconds, int decodeTokens, double decodeSeconds, long cacheBytes)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                GenerationPerformanceReport.Create(prefillTokens, prefillSeconds, decodeTokens, decodeSeconds, cacheBytes));
        }
    }

    public class CachedGenerationOutputTests
    {
        [Fact]
        public void Create_NullResult_Throws()
        {
            var performance = GenerationPerformanceReport.Create(1, 1.0, 1, 1.0, 1);

            Assert.Throws<ArgumentNullException>(() => CachedGenerationOutput.Create(null!, performance));
        }

        [Fact]
        public void Create_NullPerformance_Throws()
        {
            var result = GenerationResult.Create(new[] { 1, 2 }, GenerationFinishReason.MaxNewTokens);

            Assert.Throws<ArgumentNullException>(() => CachedGenerationOutput.Create(result, null!));
        }

        [Fact]
        public void Create_ValidArguments_ExposesBothParts()
        {
            var result = GenerationResult.Create(new[] { 1, 2 }, GenerationFinishReason.MaxNewTokens);
            var performance = GenerationPerformanceReport.Create(1, 1.0, 2, 1.0, 128);

            var output = CachedGenerationOutput.Create(result, performance);

            Assert.Same(result, output.Result);
            Assert.Same(performance, output.Performance);
        }
    }
}
