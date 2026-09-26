using System;
using Neuraval.Core.Evaluation;
using Xunit;

namespace Neuraval.Tests
{
    public class ReferenceEquivalenceThresholdsTests
    {
        [Fact]
        public void Default_MatchesRoadmapExample()
        {
            var thresholds = ReferenceEquivalenceThresholds.Default();

            Assert.Equal(0.00008f, thresholds.MaxAbsoluteError);
            Assert.Equal(0.000004f, thresholds.MeanAbsoluteError);
            Assert.Equal(0.999999f, thresholds.MinCosineSimilarity);
        }

        [Fact]
        public void Create_CustomValues_ExposesThem()
        {
            var thresholds = ReferenceEquivalenceThresholds.Create(0.01f, 0.005f, 0.99f);

            Assert.Equal(0.01f, thresholds.MaxAbsoluteError);
            Assert.Equal(0.005f, thresholds.MeanAbsoluteError);
            Assert.Equal(0.99f, thresholds.MinCosineSimilarity);
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-0.001f)]
        public void Create_MaxAbsoluteErrorNotPositive_Throws(float value)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => ReferenceEquivalenceThresholds.Create(value, 0.005f, 0.99f));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(-0.001f)]
        public void Create_MeanAbsoluteErrorNotPositive_Throws(float value)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => ReferenceEquivalenceThresholds.Create(0.01f, value, 0.99f));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(1.5f)]
        [InlineData(-0.1f)]
        public void Create_MinCosineSimilarityOutOfRange_Throws(float value)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => ReferenceEquivalenceThresholds.Create(0.01f, 0.005f, value));
        }

        [Fact]
        public void Create_MinCosineSimilarityExactlyOne_IsAllowed()
        {
            var thresholds = ReferenceEquivalenceThresholds.Create(0.01f, 0.005f, 1f);

            Assert.Equal(1f, thresholds.MinCosineSimilarity);
        }
    }
}
