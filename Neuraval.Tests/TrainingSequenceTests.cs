using System;
using Neuraval.Abstractions;
using Neuraval.Core.Training;
using Xunit;

namespace Neuraval.Tests
{
    public class TrainingSequenceTests
    {
        [Fact]
        public void Constructor_WithoutTurns_HasTurnsIsFalse()
        {
            var sequence = new TrainingSequence(new[] { 1, 2, 3 });

            Assert.False(sequence.HasTurns);
            Assert.Empty(sequence.Turns);
        }

        [Fact]
        public void Constructor_OrdersTurnsByStartIndex()
        {
            var tokens = new[] { 1, 2, 3, 4, 5, 6 };
            var turns = new[]
            {
                new TrainingTurnSpan(ChatRole.Assistant, 4, 6),
                new TrainingTurnSpan(ChatRole.User, 0, 4)
            };

            var sequence = new TrainingSequence(tokens, turns);

            Assert.Equal(ChatRole.User, sequence.Turns[0].Role);
            Assert.Equal(ChatRole.Assistant, sequence.Turns[1].Role);
        }

        [Fact]
        public void Constructor_TurnBeyondSequenceLength_Throws()
        {
            var tokens = new[] { 1, 2, 3 };
            var turns = new[] { new TrainingTurnSpan(ChatRole.Assistant, 1, 5) };

            Assert.Throws<ArgumentException>(() => new TrainingSequence(tokens, turns));
        }

        [Fact]
        public void Constructor_NullTokens_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new TrainingSequence(null!));
        }
    }
}
