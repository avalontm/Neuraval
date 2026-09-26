using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Abstractions;

namespace Neuraval.Core.Training
{
    public sealed class SupervisedFineTuningObjective : TrainingObjective
    {
        public SupervisedFineTuningObjective(IEnumerable<ChatRole>? trainableRoles = null)
        {
            TrainableRoles = trainableRoles != null
                ? new HashSet<ChatRole>(trainableRoles)
                : new HashSet<ChatRole> { ChatRole.Assistant };

            if (TrainableRoles.Count == 0)
                throw new ArgumentException("SupervisedFineTuningObjective necesita al menos un rol entrenable", nameof(trainableRoles));
        }

        public IReadOnlyCollection<ChatRole> TrainableRoles { get; }

        public override string Name => "sft";

        public override LabelMask BuildLabelMask(TrainingSequence sequence)
        {
            if (sequence == null)
                throw new ArgumentNullException(nameof(sequence));

            int length = sequence.Tokens.Length;

            if (length < 2)
                throw new ArgumentException("Se necesitan al menos 2 tokens para calcular loss de SFT", nameof(sequence));

            if (!sequence.HasTurns)
                throw new InvalidOperationException("SupervisedFineTuningObjective requiere una secuencia con turnos etiquetados por rol");

            var trainableSpans = sequence.Turns.Where(turn => TrainableRoles.Contains(turn.Role)).ToList();

            return LabelMask.FromPredicate(length, index =>
            {
                if (index < 1)
                    return false;

                return trainableSpans.Any(span => span.Contains(index));
            });
        }
    }
}
