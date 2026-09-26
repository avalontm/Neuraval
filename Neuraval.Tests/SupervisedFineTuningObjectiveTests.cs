using System;
using Neuraval.Abstractions;
using Neuraval.Core.Training;
using Xunit;

namespace Neuraval.Tests
{
    public class SupervisedFineTuningObjectiveTests
    {
        private static TrainingSequence BuildSystemUserAssistantSequence()
        {
            var tokens = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
            var turns = new[]
            {
                new TrainingTurnSpan(ChatRole.System, 0, 2),
                new TrainingTurnSpan(ChatRole.User, 2, 5),
                new TrainingTurnSpan(ChatRole.Assistant, 5, 9)
            };

            return new TrainingSequence(tokens, turns);
        }

        [Fact]
        public void BuildLabelMask_DefaultRoles_OnlyLabelsAssistantTokens()
        {
            var objective = new SupervisedFineTuningObjective();
            var sequence = BuildSystemUserAssistantSequence();

            var mask = objective.BuildLabelMask(sequence);

            for (int i = 0; i < 5; i++)
                Assert.False(mask[i]);

            for (int i = 5; i < 9; i++)
                Assert.True(mask[i]);

            Assert.Equal(4, mask.LabeledCount);
        }

        [Fact]
        public void BuildLabelMask_MultiTurnConversation_MasksEachAssistantSpanIndependently()
        {
            var tokens = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            var turns = new[]
            {
                new TrainingTurnSpan(ChatRole.User, 0, 2),
                new TrainingTurnSpan(ChatRole.Assistant, 2, 4),
                new TrainingTurnSpan(ChatRole.User, 4, 7),
                new TrainingTurnSpan(ChatRole.Assistant, 7, 10)
            };
            var sequence = new TrainingSequence(tokens, turns);
            var objective = new SupervisedFineTuningObjective();

            var mask = objective.BuildLabelMask(sequence);

            Assert.True(mask[2]);
            Assert.True(mask[3]);
            Assert.False(mask[4]);
            Assert.False(mask[5]);
            Assert.False(mask[6]);
            Assert.True(mask[7]);
            Assert.True(mask[9]);
            Assert.Equal(5, mask.LabeledCount);
        }

        [Fact]
        public void BuildLabelMask_CustomTrainableRoles_LabelsConfiguredRoles()
        {
            var objective = new SupervisedFineTuningObjective(new[] { ChatRole.Tool, ChatRole.Assistant });
            var tokens = new[] { 1, 2, 3, 4, 5, 6 };
            var turns = new[]
            {
                new TrainingTurnSpan(ChatRole.User, 0, 2),
                new TrainingTurnSpan(ChatRole.Tool, 2, 4),
                new TrainingTurnSpan(ChatRole.Assistant, 4, 6)
            };
            var sequence = new TrainingSequence(tokens, turns);

            var mask = objective.BuildLabelMask(sequence);

            Assert.True(mask[2]);
            Assert.True(mask[3]);
            Assert.True(mask[4]);
            Assert.True(mask[5]);
        }

        [Fact]
        public void BuildLabelMask_SequenceWithoutTurns_Throws()
        {
            var objective = new SupervisedFineTuningObjective();
            var sequence = new TrainingSequence(new[] { 1, 2, 3 });

            Assert.Throws<InvalidOperationException>(() => objective.BuildLabelMask(sequence));
        }

        [Fact]
        public void Constructor_EmptyRoleSet_Throws()
        {
            Assert.Throws<ArgumentException>(() => new SupervisedFineTuningObjective(Array.Empty<ChatRole>()));
        }

        [Fact]
        public void Name_IsSft()
        {
            Assert.Equal("sft", new SupervisedFineTuningObjective().Name);
        }
    }
}
