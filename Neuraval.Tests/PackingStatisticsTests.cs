using System;
using System.Collections.Generic;
using Neuraval.Core.Training;
using Neuraval.Core.Training.Packing;
using Xunit;

namespace Neuraval.Tests
{
    public class PackingStatisticsTests
    {
        [Fact]
        public void From_ComputesAggregatedTokenCounts()
        {
            var packer = new SequencePacker(new CausalLanguageModelingObjective(), maxSequenceLength: 6, padToken: 0);
            var sequences = new List<TrainingSequence>
            {
                new TrainingSequence(new[] { 1, 2, 3 }),
                new TrainingSequence(new[] { 4, 5 }),
                new TrainingSequence(new[] { 6, 7, 8, 9 })
            };

            var packedSequences = packer.Pack(sequences);
            var statistics = PackingStatistics.From(packedSequences);

            Assert.Equal(12, statistics.TokensSeen);
            Assert.Equal(3, statistics.PaddingTokens);
            Assert.Equal(9, statistics.UsefulTokens);
            Assert.Equal(0.75, statistics.PackingEfficiency, 3);
        }

        [Fact]
        public void From_EmptyPackedSequences_ReturnsZeroedStatisticsWithoutDividingByZero()
        {
            var statistics = PackingStatistics.From(new List<PackedSequence>());

            Assert.Equal(0, statistics.TokensSeen);
            Assert.Equal(0, statistics.PaddingTokens);
            Assert.Equal(0, statistics.UsefulTokens);
            Assert.Equal(0, statistics.PackingEfficiency);
        }

        [Fact]
        public void From_NullPackedSequences_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => PackingStatistics.From(null!));
        }

        [Fact]
        public void From_NoPaddingAnywhere_EfficiencyIsOne()
        {
            var packer = new SequencePacker(new CausalLanguageModelingObjective(), maxSequenceLength: 3, padToken: 0);
            var sequences = new List<TrainingSequence> { new TrainingSequence(new[] { 1, 2, 3 }) };

            var packedSequences = packer.Pack(sequences);
            var statistics = PackingStatistics.From(packedSequences);

            Assert.Equal(1.0, statistics.PackingEfficiency, 3);
        }
    }
}
