using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Core.Training
{
    public sealed class TrainingSequence
    {
        public TrainingSequence(int[] tokens, IReadOnlyList<TrainingTurnSpan>? turns = null)
        {
            Tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));

            var resolvedTurns = turns ?? Array.Empty<TrainingTurnSpan>();

            foreach (var turn in resolvedTurns)
            {
                if (turn.EndIndex > Tokens.Length)
                {
                    throw new ArgumentException(
                        $"El turno {turn.Role} [{turn.StartIndex}, {turn.EndIndex}) excede la longitud de la secuencia ({Tokens.Length} tokens)",
                        nameof(turns));
                }
            }

            Turns = resolvedTurns.OrderBy(turn => turn.StartIndex).ToList();
        }

        public int[] Tokens { get; }

        public IReadOnlyList<TrainingTurnSpan> Turns { get; }

        public bool HasTurns => Turns.Count > 0;
    }
}
