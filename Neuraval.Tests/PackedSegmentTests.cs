using System;
using Neuraval.Core.Training.Packing;
using Xunit;

namespace Neuraval.Tests
{
    public class PackedSegmentTests
    {
        [Fact]
        public void Constructor_ExposesFieldsAndComputesLength()
        {
            var segment = new PackedSegment(2, 3, 7);

            Assert.Equal(2, segment.SequenceIndex);
            Assert.Equal(3, segment.StartIndex);
            Assert.Equal(7, segment.EndIndex);
            Assert.Equal(4, segment.Length);
        }

        [Fact]
        public void Constructor_NegativeSequenceIndex_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new PackedSegment(-1, 0, 2));
        }

        [Fact]
        public void Constructor_NegativeStartIndex_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new PackedSegment(0, -1, 2));
        }

        [Fact]
        public void Constructor_EndIndexNotGreaterThanStartIndex_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new PackedSegment(0, 3, 3));
        }
    }
}
