using System;
using Neuraval.Abstractions;

namespace Neuraval.Core.Training
{
    public sealed class TrainingTurnSpan
    {
        public TrainingTurnSpan(ChatRole role, int startIndex, int endIndex)
        {
            if (startIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(startIndex), "startIndex no puede ser negativo");

            if (endIndex <= startIndex)
                throw new ArgumentOutOfRangeException(nameof(endIndex), "endIndex debe ser mayor que startIndex");

            Role = role;
            StartIndex = startIndex;
            EndIndex = endIndex;
        }

        public ChatRole Role { get; }

        public int StartIndex { get; }

        public int EndIndex { get; }

        public bool Contains(int tokenIndex)
        {
            return tokenIndex >= StartIndex && tokenIndex < EndIndex;
        }
    }
}
