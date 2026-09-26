using System;

namespace Neuraval.Core.Training.Packing
{
    public sealed class PackedSegment
    {
        public PackedSegment(int sequenceIndex, int startIndex, int endIndex)
        {
            if (sequenceIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(sequenceIndex), "sequenceIndex no puede ser negativo");

            if (startIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(startIndex), "startIndex no puede ser negativo");

            if (endIndex <= startIndex)
                throw new ArgumentOutOfRangeException(nameof(endIndex), "endIndex debe ser mayor que startIndex");

            SequenceIndex = sequenceIndex;
            StartIndex = startIndex;
            EndIndex = endIndex;
        }

        public int SequenceIndex { get; }

        public int StartIndex { get; }

        public int EndIndex { get; }

        public int Length => EndIndex - StartIndex;
    }
}
