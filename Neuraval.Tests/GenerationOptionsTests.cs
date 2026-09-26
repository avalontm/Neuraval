using System;
using Neuraval.Core.Generation;
using Neuraval.Core.Serialization.ModelExport;
using Xunit;

namespace Neuraval.Tests
{
    public class GenerationOptionsTests
    {
        [Fact]
        public void Create_ValidValues_SetsAllProperties()
        {
            var stopTokenIds = new[] { 2, 7 };

            var options = GenerationOptions.Create(false, 0.8f, 0.95f, 20, 1.2f, 64, stopTokenIds, seed: 5);

            Assert.False(options.Greedy);
            Assert.Equal(0.8f, options.Temperature);
            Assert.Equal(0.95f, options.TopP);
            Assert.Equal(20, options.TopK);
            Assert.Equal(1.2f, options.RepetitionPenalty);
            Assert.Equal(64, options.MaxNewTokens);
            Assert.Equal(stopTokenIds, options.StopTokenIds);
            Assert.Equal(5, options.Seed);
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-1f)]
        public void Create_InvalidTemperature_Throws(float temperature)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                GenerationOptions.Create(false, temperature, 0.9f, 40, 1f, 10, Array.Empty<int>()));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-0.1f)]
        [InlineData(1.1f)]
        public void Create_InvalidTopP_Throws(float topP)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                GenerationOptions.Create(false, 0.7f, topP, 40, 1f, 10, Array.Empty<int>()));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-5)]
        public void Create_InvalidTopK_Throws(int topK)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                GenerationOptions.Create(false, 0.7f, 0.9f, topK, 1f, 10, Array.Empty<int>()));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-1f)]
        public void Create_InvalidRepetitionPenalty_Throws(float penalty)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                GenerationOptions.Create(false, 0.7f, 0.9f, 40, penalty, 10, Array.Empty<int>()));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-3)]
        public void Create_InvalidMaxNewTokens_Throws(int maxNewTokens)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                GenerationOptions.Create(false, 0.7f, 0.9f, 40, 1f, maxNewTokens, Array.Empty<int>()));
        }

        [Fact]
        public void Create_NullStopTokenIds_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                GenerationOptions.Create(false, 0.7f, 0.9f, 40, 1f, 10, null!));
        }

        [Fact]
        public void CreateGreedy_SetsGreedyTrueAndPassthroughValues()
        {
            var stopTokenIds = new[] { 3 };

            var options = GenerationOptions.CreateGreedy(50, stopTokenIds);

            Assert.True(options.Greedy);
            Assert.Equal(50, options.MaxNewTokens);
            Assert.Equal(stopTokenIds, options.StopTokenIds);
        }

        [Fact]
        public void Default_MatchesRoadmapValues()
        {
            var options = GenerationOptions.Default(Array.Empty<int>());

            Assert.False(options.Greedy);
            Assert.Equal(0.7f, options.Temperature);
            Assert.Equal(0.9f, options.TopP);
            Assert.Equal(40, options.TopK);
            Assert.Equal(256, options.MaxNewTokens);
        }

        [Fact]
        public void FromExportConfig_CopiesTemperatureTopPTopKAndMaxNewTokens()
        {
            var exportConfig = GenerationConfigOptions.Create(0.6f, 0.85f, 30, 128);

            var options = GenerationOptions.FromExportConfig(exportConfig, repetitionPenalty: 1.3f, stopTokenIds: new[] { 1 });

            Assert.Equal(0.6f, options.Temperature);
            Assert.Equal(0.85f, options.TopP);
            Assert.Equal(30, options.TopK);
            Assert.Equal(128, options.MaxNewTokens);
            Assert.Equal(1.3f, options.RepetitionPenalty);
            Assert.False(options.Greedy);
        }

        [Fact]
        public void FromExportConfig_NullExportConfig_Throws()
        {
            Assert.Throws<ArgumentNullException>(() =>
                GenerationOptions.FromExportConfig(null!, 1f, Array.Empty<int>()));
        }
    }
}
