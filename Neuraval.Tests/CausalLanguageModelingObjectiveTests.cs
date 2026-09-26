using System;
using Neuraval.Abstractions;
using Neuraval.Core.Training;
using Xunit;

namespace Neuraval.Tests
{
    public class CausalLanguageModelingObjectiveTests
    {
        [Fact]
        public void BuildLabelMask_LabelsEveryTokenExceptTheFirst()
        {
            var objective = new CausalLanguageModelingObjective();
            var sequence = new TrainingSequence(new[] { 10, 11, 12, 13, 14 });

            var mask = objective.BuildLabelMask(sequence);

            Assert.False(mask[0]);
            Assert.True(mask[1]);
            Assert.True(mask[4]);
            Assert.Equal(4, mask.LabeledCount);
        }

        [Fact]
        public void BuildLabelMask_IgnoresTurnBoundaries()
        {
            var objective = new CausalLanguageModelingObjective();
            var tokens = new[] { 1, 2, 3, 4 };
            var turns = new[] { new TrainingTurnSpan(ChatRole.User, 0, 4) };
            var sequence = new TrainingSequence(tokens, turns);

            var mask = objective.BuildLabelMask(sequence);

            Assert.Equal(3, mask.LabeledCount);
        }

        [Fact]
        public void BuildLabelMask_SequenceShorterThanTwoTokens_Throws()
        {
            var objective = new CausalLanguageModelingObjective();
            var sequence = new TrainingSequence(new[] { 1 });

            Assert.Throws<ArgumentException>(() => objective.BuildLabelMask(sequence));
        }

        [Fact]
        public void BuildLabelMask_NullSequence_Throws()
        {
            var objective = new CausalLanguageModelingObjective();

            Assert.Throws<ArgumentNullException>(() => objective.BuildLabelMask(null!));
        }

        [Fact]
        public void Name_IsCausalLm()
        {
            Assert.Equal("causal-lm", new CausalLanguageModelingObjective().Name);
        }
    }
}
