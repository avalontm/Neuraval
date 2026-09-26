using System.Collections.Generic;

namespace Neuraval.Core.Serialization.WeightLoading
{
    public sealed class WeightManifestReport
    {
        public IReadOnlyList<string> MissingNames { get; }

        public IReadOnlyList<string> UnexpectedNames { get; }

        public bool IsComplete => MissingNames.Count == 0;

        public WeightManifestReport(IReadOnlyList<string> missingNames, IReadOnlyList<string> unexpectedNames)
        {
            MissingNames = missingNames;
            UnexpectedNames = unexpectedNames;
        }
    }
}
