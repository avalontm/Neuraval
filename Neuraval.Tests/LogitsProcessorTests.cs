using System;
using Neuraval.Core.Generation;
using Xunit;

namespace Neuraval.Tests
{
    public class LogitsProcessorTests
    {
        [Fact]
        public void ApplyRepetitionPenalty_NoGeneratedTokens_NoChange()
        {
            var logits = new float[] { 1f, 2f, 3f };

            LogitsProcessor.ApplyRepetitionPenalty(logits, Array.Empty<int>(), 1.5f);

            Assert.Equal(new float[] { 1f, 2f, 3f }, logits);
        }

        [Fact]
        public void ApplyRepetitionPenalty_PenaltyOne_NoChange()
        {
            var logits = new float[] { 1f, 2f, 3f };

            LogitsProcessor.ApplyRepetitionPenalty(logits, new[] { 0, 1 }, 1f);

            Assert.Equal(new float[] { 1f, 2f, 3f }, logits);
        }

        [Fact]
        public void ApplyRepetitionPenalty_PositiveLogit_DividesByPenalty()
        {
            var logits = new float[] { 4f, 2f };

            LogitsProcessor.ApplyRepetitionPenalty(logits, new[] { 0 }, 2f);

            Assert.Equal(2f, logits[0]);
            Assert.Equal(2f, logits[1]);
        }

        [Fact]
        public void ApplyRepetitionPenalty_NegativeLogit_MultipliesByPenalty()
        {
            var logits = new float[] { -4f };

            LogitsProcessor.ApplyRepetitionPenalty(logits, new[] { 0 }, 2f);

            Assert.Equal(-8f, logits[0]);
        }

        [Fact]
        public void ApplyRepetitionPenalty_DuplicateTokenIds_AppliedOnce()
        {
            var logits = new float[] { 8f };

            LogitsProcessor.ApplyRepetitionPenalty(logits, new[] { 0, 0, 0 }, 2f);

            Assert.Equal(4f, logits[0]);
        }

        [Fact]
        public void ApplyRepetitionPenalty_InvalidPenalty_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                LogitsProcessor.ApplyRepetitionPenalty(new float[] { 1f }, new[] { 0 }, 0f));
        }

        [Fact]
        public void ApplyTemperature_TemperatureOne_NoChange()
        {
            var logits = new float[] { 1f, -2f };

            LogitsProcessor.ApplyTemperature(logits, 1f);

            Assert.Equal(new float[] { 1f, -2f }, logits);
        }

        [Fact]
        public void ApplyTemperature_ScalesLogitsByDivision()
        {
            var logits = new float[] { 4f, -2f };

            LogitsProcessor.ApplyTemperature(logits, 2f);

            Assert.Equal(new float[] { 2f, -1f }, logits);
        }

        [Fact]
        public void ApplyTemperature_InvalidTemperature_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                LogitsProcessor.ApplyTemperature(new float[] { 1f }, 0f));
        }

        [Fact]
        public void FilterTopK_KeepsOnlyTopKValues()
        {
            var logits = new float[] { 1f, 5f, 3f, 2f, 4f };

            LogitsProcessor.FilterTopK(logits, 2);

            Assert.Equal(float.NegativeInfinity, logits[0]);
            Assert.Equal(5f, logits[1]);
            Assert.Equal(float.NegativeInfinity, logits[2]);
            Assert.Equal(float.NegativeInfinity, logits[3]);
            Assert.Equal(4f, logits[4]);
        }

        [Fact]
        public void FilterTopK_TopKGreaterOrEqualToLength_NoChange()
        {
            var logits = new float[] { 1f, 2f, 3f };

            LogitsProcessor.FilterTopK(logits, 10);

            Assert.Equal(new float[] { 1f, 2f, 3f }, logits);
        }

        [Fact]
        public void FilterTopK_InvalidTopK_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                LogitsProcessor.FilterTopK(new float[] { 1f }, 0));
        }

        [Fact]
        public void FilterTopP_KeepsNucleusAndRenormalizes()
        {
            var probabilities = new float[] { 0.5f, 0.3f, 0.1f, 0.1f };

            LogitsProcessor.FilterTopP(probabilities, 0.8f);

            Assert.Equal(0f, probabilities[2]);
            Assert.Equal(0f, probabilities[3]);
            Assert.True(probabilities[0] > 0f);
            Assert.True(probabilities[1] > 0f);

            float sum = probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3];
            Assert.Equal(1f, sum, 5);
        }

        [Fact]
        public void FilterTopP_TopPOne_NoChange()
        {
            var probabilities = new float[] { 0.5f, 0.3f, 0.2f };

            LogitsProcessor.FilterTopP(probabilities, 1f);

            Assert.Equal(new float[] { 0.5f, 0.3f, 0.2f }, probabilities);
        }

        [Fact]
        public void FilterTopP_InvalidTopP_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                LogitsProcessor.FilterTopP(new float[] { 1f }, 0f));
        }
    }
}
