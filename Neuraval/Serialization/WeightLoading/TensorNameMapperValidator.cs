using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.SafeTensors;

namespace Neuraval.Core.Serialization.WeightLoading
{
    public static class TensorNameMapperValidator
    {
        public static WeightManifestReport Validate(TransformerConfig config, IEnumerable<string> availableNames)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            if (availableNames == null)
                throw new ArgumentNullException(nameof(availableNames));

            var expected = new HashSet<string>(TensorNameMapper.AllExpectedNames(config));
            var available = new HashSet<string>(availableNames);

            var missing = expected.Except(available).OrderBy(name => name, StringComparer.Ordinal).ToList();
            var unexpected = available.Except(expected).OrderBy(name => name, StringComparer.Ordinal).ToList();

            return new WeightManifestReport(missing, unexpected);
        }

        public static WeightManifestReport Validate(TransformerConfig config, SafeTensorsFile file)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));

            return Validate(config, file.Tensors.Select(tensor => tensor.Name));
        }
    }
}
