using System;
using Neuraval.Core.Serialization.ModelExport;
using Xunit;

namespace Neuraval.Tests
{
    public class GenerationConfigOptionsTests
    {
        [Fact]
        public void Default_MatchesRoadmapValues()
        {
            var options = GenerationConfigOptions.Default();

            Assert.Equal(0.7f, options.Temperature);
            Assert.Equal(0.9f, options.TopP);
            Assert.Equal(40, options.TopK);
            Assert.Equal(256, options.MaxNewTokens);
        }

        [Fact]
        public void Create_CustomValues_ExposesThem()
        {
            var options = GenerationConfigOptions.Create(1.2f, 0.95f, 50, 512);

            Assert.Equal(1.2f, options.Temperature);
            Assert.Equal(0.95f, options.TopP);
            Assert.Equal(50, options.TopK);
            Assert.Equal(512, options.MaxNewTokens);
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-1f)]
        public void Create_TemperatureNotPositive_Throws(float temperature)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => GenerationConfigOptions.Create(temperature, 0.9f, 40, 256));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(1.5f)]
        public void Create_TopPOutOfRange_Throws(float topP)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => GenerationConfigOptions.Create(0.7f, topP, 40, 256));
        }

        [Fact]
        public void Create_TopKNotPositive_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => GenerationConfigOptions.Create(0.7f, 0.9f, 0, 256));
        }

        [Fact]
        public void Create_MaxNewTokensNotPositive_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => GenerationConfigOptions.Create(0.7f, 0.9f, 40, 0));
        }
    }
}
