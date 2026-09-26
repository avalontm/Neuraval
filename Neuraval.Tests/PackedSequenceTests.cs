using System;
using System.Collections.Generic;
using Neuraval.Core.Training;
using Neuraval.Core.Training.Packing;
using Xunit;

namespace Neuraval.Tests
{
    public class PackedSequenceTests
    {
        [Fact]
        public void Constructor_ComputesUsefulTokenCount()
        {
            var tokens = new[] { 1, 2, 3, 0, 0 };
            var mask = LabelMask.AllLabeled(5);
            var segments = new List<PackedSegment> { new PackedSegment(0, 0, 3) };

            var packedSequence = new PackedSequence(tokens, mask, segments, 2);

            Assert.Equal(5, packedSequence.Capacity);
            Assert.Equal(2, packedSequence.PaddingTokenCount);
            Assert.Equal(3, packedSequence.UsefulTokenCount);
        }

        [Fact]
        public void Constructor_LabelMaskLengthMismatch_Throws()
        {
            var tokens = new[] { 1, 2, 3 };
            var mask = LabelMask.AllLabeled(2);
            var segments = new List<PackedSegment> { new PackedSegment(0, 0, 3) };

            Assert.Throws<ArgumentException>(() => new PackedSequence(tokens, mask, segments, 0));
        }

        [Fact]
        public void Constructor_PaddingTokenCountOutOfRange_Throws()
        {
            var tokens = new[] { 1, 2, 3 };
            var mask = LabelMask.AllLabeled(3);
            var segments = new List<PackedSegment> { new PackedSegment(0, 0, 3) };

            Assert.Throws<ArgumentOutOfRangeException>(() => new PackedSequence(tokens, mask, segments, 4));
        }

        [Fact]
        public void Constructor_NullTokens_Throws()
        {
            var mask = LabelMask.AllLabeled(3);
            var segments = new List<PackedSegment>();

            Assert.Throws<ArgumentNullException>(() => new PackedSequence(null!, mask, segments, 0));
        }
    }
}
