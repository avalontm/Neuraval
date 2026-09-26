using System;

namespace Neuraval.Core.Training
{
    public sealed class CausalLanguageModelingObjective : TrainingObjective
    {
        public override string Name => "causal-lm";

        public override LabelMask BuildLabelMask(TrainingSequence sequence)
        {
            if (sequence == null)
                throw new ArgumentNullException(nameof(sequence));

            int length = sequence.Tokens.Length;

            if (length < 2)
                throw new ArgumentException("Se necesitan al menos 2 tokens para calcular loss causal", nameof(sequence));

            return LabelMask.FromPredicate(length, index => index >= 1);
        }
    }
}
