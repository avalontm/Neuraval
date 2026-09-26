using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.SafeTensors;
using Neuraval.Core.Serialization.WeightLoading;

namespace Neuraval.Core.Serialization.Gguf
{
    public sealed class GgufLoadResult
    {
        public TransformerConfig Config { get; }

        public IReadOnlyList<SafeTensorsEntry> Weights { get; }

        public WeightManifestReport Report { get; }

        public GgufLoadResult(TransformerConfig config, IReadOnlyList<SafeTensorsEntry> weights, WeightManifestReport report)
        {
            Config = config;
            Weights = weights;
            Report = report;
        }
    }

    public static class GgufModelLoader
    {
        public static TransformerConfig LoadConfig(GgufFile file)
        {
            return GgufConfigMapper.MapConfig(file);
        }

        public static IReadOnlyList<SafeTensorsEntry> LoadWeights(GgufFile file)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));

            var entries = new List<SafeTensorsEntry>();

            foreach (var tensor in file.Tensors)
            {
                if (!GgufTensorNameMapper.TryMapToNeuravalName(tensor.Name, out var neuravalName))
                    continue;

                entries.Add(new SafeTensorsEntry(neuravalName!, SafeTensorsDType.F32, tensor.Shape, tensor.Data));
            }

            return entries;
        }

        public static WeightManifestReport ValidateWeights(TransformerConfig config, IReadOnlyList<SafeTensorsEntry> weights)
        {
            return TensorNameMapperValidator.Validate(config, weights.Select(entry => entry.Name));
        }

        public static GgufLoadResult Load(string filePath)
        {
            var file = GgufReader.Read(filePath);
            var config = LoadConfig(file);
            var weights = LoadWeights(file);
            var report = ValidateWeights(config, weights);

            return new GgufLoadResult(config, weights, report);
        }

        public static GgufLoadResult Load(GgufFile file)
        {
            var config = LoadConfig(file);
            var weights = LoadWeights(file);
            var report = ValidateWeights(config, weights);

            return new GgufLoadResult(config, weights, report);
        }
    }
}
