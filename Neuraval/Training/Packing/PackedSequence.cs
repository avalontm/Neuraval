using System;
using System.Collections.Generic;

namespace Neuraval.Core.Training.Packing
{
    public sealed class PackedSequence
    {
        public PackedSequence(int[] tokens, LabelMask labelMask, IReadOnlyList<PackedSegment> segments, int paddingTokenCount)
        {
            Tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
            LabelMask = labelMask ?? throw new ArgumentNullException(nameof(labelMask));
            Segments = segments ?? throw new ArgumentNullException(nameof(segments));

            if (LabelMask.Length != Tokens.Length)
                throw new ArgumentException("LabelMask debe tener la misma longitud que Tokens", nameof(labelMask));

            if (paddingTokenCount < 0 || paddingTokenCount > Tokens.Length)
                throw new ArgumentOutOfRangeException(nameof(paddingTokenCount), "paddingTokenCount fuera de rango");

            PaddingTokenCount = paddingTokenCount;
        }

        public int[] Tokens { get; }

        public LabelMask LabelMask { get; }

        public IReadOnlyList<PackedSegment> Segments { get; }

        public int PaddingTokenCount { get; }

        public int Capacity => Tokens.Length;

        public int UsefulTokenCount => Capacity - PaddingTokenCount;
    }
}
