using System.Collections.Generic;

namespace Neuraval.Core.Evaluation
{
    public sealed class EquivalenceVerdict
    {
        public bool IsEquivalent { get; }

        public IReadOnlyList<string> FailureReasons { get; }

        public EquivalenceVerdict(bool isEquivalent, IReadOnlyList<string> failureReasons)
        {
            IsEquivalent = isEquivalent;
            FailureReasons = failureReasons;
        }
    }
}
