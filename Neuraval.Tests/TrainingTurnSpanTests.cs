using System;
using Neuraval.Abstractions;
using Neuraval.Core.Training;
using Xunit;

namespace Neuraval.Tests
{
    public class TrainingTurnSpanTests
    {
        [Fact]
        public void Contains_ReturnsTrueOnlyWithinRange()
        {
            var span = new TrainingTurnSpan(ChatRole.Assistant, 3, 6);

            Assert.False(span.Contains(2));
            Assert.True(span.Contains(3));
            Assert.True(span.Contains(5));
            Assert.False(span.Contains(6));
        }

        [Fact]
        public void Constructor_ExposesRoleAndBounds()
        {
            var span = new TrainingTurnSpan(ChatRole.User, 1, 4);

            Assert.Equal(ChatRole.User, span.Role);
            Assert.Equal(1, span.StartIndex);
            Assert.Equal(4, span.EndIndex);
        }

        [Fact]
        public void Constructor_NegativeStartIndex_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingTurnSpan(ChatRole.User, -1, 2));
        }

        [Fact]
        public void Constructor_EndIndexNotGreaterThanStartIndex_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new TrainingTurnSpan(ChatRole.User, 4, 4));
        }
    }
}
