using System;
using System.Collections.Generic;
using Neuraval.Abstractions;
using Neuraval.Core.Training;
using Neuraval.Core.Training.Packing;
using Xunit;

namespace Neuraval.Tests
{
    public class SequencePackerTests
    {
        private static List<TrainingSequence> BuildThreeExamples()
        {
            return new List<TrainingSequence>
            {
                new TrainingSequence(new[] { 1, 2, 3 }),
                new TrainingSequence(new[] { 4, 5 }),
                new TrainingSequence(new[] { 6, 7, 8, 9 })
            };
        }

        [Fact]
        public void Pack_FitsMultipleExamplesInOnePackedSequence()
        {
            var packer = new SequencePacker(new CausalLanguageModelingObjective(), maxSequenceLength: 6, padToken: 0);

            var packedSequences = packer.Pack(BuildThreeExamples());

            Assert.Equal(2, packedSequences.Count);

            var first = packedSequences[0];
            Assert.Equal(new[] { 1, 2, 3, 4, 5, 0 }, first.Tokens);
            Assert.Equal(1, first.PaddingTokenCount);
            Assert.Equal(2, first.Segments.Count);
            Assert.Equal(0, first.Segments[0].SequenceIndex);
            Assert.Equal(0, first.Segments[0].StartIndex);
            Assert.Equal(3, first.Segments[0].EndIndex);
            Assert.Equal(1, first.Segments[1].SequenceIndex);
            Assert.Equal(3, first.Segments[1].StartIndex);
            Assert.Equal(5, first.Segments[1].EndIndex);
        }

        [Fact]
        public void Pack_StartsNewPackedSequenceWhenNextExampleDoesNotFit()
        {
            var packer = new SequencePacker(new CausalLanguageModelingObjective(), maxSequenceLength: 6, padToken: 0);

            var packedSequences = packer.Pack(BuildThreeExamples());

            var second = packedSequences[1];
            Assert.Equal(new[] { 6, 7, 8, 9, 0, 0 }, second.Tokens);
            Assert.Equal(2, second.PaddingTokenCount);
            Assert.Single(second.Segments);
            Assert.Equal(2, second.Segments[0].SequenceIndex);
        }

        [Fact]
        public void Pack_LabelMaskCombinesPerExampleMasksAndMasksPadding()
        {
            var packer = new SequencePacker(new CausalLanguageModelingObjective(), maxSequenceLength: 6, padToken: 0);

            var packedSequences = packer.Pack(BuildThreeExamples());

            var expected = new[] { false, true, true, false, true, false };
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], packedSequences[0].LabelMask[i]);
        }

        [Fact]
        public void Pack_UsesTrainableRolesFromObjective()
        {
            var objective = new SupervisedFineTuningObjective();
            var tokens = new[] { 1, 2, 3, 4 };
            var turns = new[]
            {
                new TrainingTurnSpan(ChatRole.User, 0, 2),
                new TrainingTurnSpan(ChatRole.Assistant, 2, 4)
            };
            var sequences = new List<TrainingSequence> { new TrainingSequence(tokens, turns) };

            var packer = new SequencePacker(objective, maxSequenceLength: 4, padToken: 0);
            var packedSequences = packer.Pack(sequences);

            var mask = packedSequences[0].LabelMask;
            Assert.False(mask[0]);
            Assert.False(mask[1]);
            Assert.True(mask[2]);
            Assert.True(mask[3]);
        }

        [Fact]
        public void Pack_ExampleExceedingMaxSequenceLength_Throws()
        {
            var packer = new SequencePacker(new CausalLanguageModelingObjective(), maxSequenceLength: 3, padToken: 0);
            var sequences = new List<TrainingSequence> { new TrainingSequence(new[] { 1, 2, 3, 4 }) };

            Assert.Throws<ArgumentException>(() => packer.Pack(sequences));
        }

        [Fact]
        public void Pack_EmptyInput_ReturnsEmptyList()
        {
            var packer = new SequencePacker(new CausalLanguageModelingObjective(), maxSequenceLength: 4, padToken: 0);

            var packedSequences = packer.Pack(new List<TrainingSequence>());

            Assert.Empty(packedSequences);
        }

        [Fact]
        public void Pack_ExactFit_NoPaddingAdded()
        {
            var packer = new SequencePacker(new CausalLanguageModelingObjective(), maxSequenceLength: 3, padToken: 0);
            var sequences = new List<TrainingSequence> { new TrainingSequence(new[] { 1, 2, 3 }) };

            var packedSequences = packer.Pack(sequences);

            Assert.Equal(0, packedSequences[0].PaddingTokenCount);
        }

        [Fact]
        public void Constructor_MaxSequenceLengthBelowTwo_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new SequencePacker(new CausalLanguageModelingObjective(), 1));
        }

        [Fact]
        public void Constructor_NullObjective_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => new SequencePacker(null!, 4));
        }
    }
}
