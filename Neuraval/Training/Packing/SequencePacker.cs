using System;
using System.Collections.Generic;

namespace Neuraval.Core.Training.Packing
{
    public sealed class SequencePacker
    {
        private readonly TrainingObjective _objective;
        private readonly int _maxSequenceLength;
        private readonly int _padToken;

        public SequencePacker(TrainingObjective objective, int maxSequenceLength, int padToken = 0)
        {
            _objective = objective ?? throw new ArgumentNullException(nameof(objective));

            if (maxSequenceLength < 2)
                throw new ArgumentOutOfRangeException(nameof(maxSequenceLength), "maxSequenceLength debe ser al menos 2");

            _maxSequenceLength = maxSequenceLength;
            _padToken = padToken;
        }

        public IReadOnlyList<PackedSequence> Pack(IReadOnlyList<TrainingSequence> sequences)
        {
            if (sequences == null)
                throw new ArgumentNullException(nameof(sequences));

            var packed = new List<PackedSequence>();

            var currentTokens = new List<int>();
            var currentMaskFlags = new List<bool>();
            var currentSegments = new List<PackedSegment>();

            for (int sequenceIndex = 0; sequenceIndex < sequences.Count; sequenceIndex++)
            {
                var sequence = sequences[sequenceIndex];

                if (sequence.Tokens.Length > _maxSequenceLength)
                {
                    throw new ArgumentException(
                        $"La secuencia {sequenceIndex} tiene {sequence.Tokens.Length} tokens y excede maxSequenceLength={_maxSequenceLength}",
                        nameof(sequences));
                }

                if (currentTokens.Count > 0 && currentTokens.Count + sequence.Tokens.Length > _maxSequenceLength)
                {
                    packed.Add(BuildPackedSequence(currentTokens, currentMaskFlags, currentSegments));
                    currentTokens.Clear();
                    currentMaskFlags.Clear();
                    currentSegments.Clear();
                }

                var mask = _objective.BuildLabelMask(sequence);
                int start = currentTokens.Count;

                currentTokens.AddRange(sequence.Tokens);

                for (int i = 0; i < sequence.Tokens.Length; i++)
                {
                    currentMaskFlags.Add(mask[i]);
                }

                currentSegments.Add(new PackedSegment(sequenceIndex, start, start + sequence.Tokens.Length));
            }

            if (currentTokens.Count > 0)
            {
                packed.Add(BuildPackedSequence(currentTokens, currentMaskFlags, currentSegments));
            }

            return packed;
        }

        private PackedSequence BuildPackedSequence(List<int> tokens, List<bool> maskFlags, List<PackedSegment> segments)
        {
            int paddingCount = _maxSequenceLength - tokens.Count;

            var paddedTokens = new int[_maxSequenceLength];
            var paddedFlags = new bool[_maxSequenceLength];

            for (int i = 0; i < tokens.Count; i++)
            {
                paddedTokens[i] = tokens[i];
                paddedFlags[i] = maskFlags[i];
            }

            for (int i = tokens.Count; i < _maxSequenceLength; i++)
            {
                paddedTokens[i] = _padToken;
                paddedFlags[i] = false;
            }

            return new PackedSequence(paddedTokens, new LabelMask(paddedFlags), new List<PackedSegment>(segments), paddingCount);
        }
    }
}
