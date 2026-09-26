using System;
using Neuraval.Core.Training;
using Xunit;

namespace Neuraval.Tests
{
    public class LabelMaskTests
    {
        [Fact]
        public void AllLabeled_MarksEveryPositionAsLabeled()
        {
            var mask = LabelMask.AllLabeled(5);

            Assert.Equal(5, mask.Length);
            Assert.Equal(5, mask.LabeledCount);

            for (int i = 0; i < mask.Length; i++)
                Assert.True(mask[i]);
        }

        [Fact]
        public void AllUnlabeled_MarksEveryPositionAsUnlabeled()
        {
            var mask = LabelMask.AllUnlabeled(4);

            Assert.Equal(4, mask.Length);
            Assert.Equal(0, mask.LabeledCount);
        }

        [Fact]
        public void FromPredicate_AppliesPredicatePerIndex()
        {
            var mask = LabelMask.FromPredicate(6, index => index % 2 == 0);

            Assert.True(mask[0]);
            Assert.False(mask[1]);
            Assert.True(mask[2]);
            Assert.Equal(3, mask.LabeledCount);
        }

        [Fact]
        public void ToArray_ReturnsIndependentCopy()
        {
            var mask = LabelMask.AllLabeled(3);
            var array = mask.ToArray();
            array[0] = false;

            Assert.True(mask[0]);
        }

        [Fact]
        public void Constructor_NullFlags_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new LabelMask(null!));
        }

        [Fact]
        public void FromPredicate_NegativeLength_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => LabelMask.FromPredicate(-1, _ => true));
        }

        [Fact]
        public void FromPredicate_NullPredicate_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => LabelMask.FromPredicate(3, null!));
        }
    }
}
