using System;
using System.IO;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.SafeTensors;

namespace Neuraval.Core.Serialization.ModelExport
{
    public sealed class ModernModelDirectoryContent
    {
        public TransformerConfig Config { get; }

        public SafeTensorsFile Weights { get; }

        public GenerationConfigOptions GenerationConfig { get; }

        public ModernModelDirectoryContent(TransformerConfig config, SafeTensorsFile weights, GenerationConfigOptions generationConfig)
        {
            Config = config;
            Weights = weights;
            GenerationConfig = generationConfig;
        }
    }

    public static class ModernModelDirectoryReader
    {
        public static ModernModelDirectoryContent Read(string directoryPath)
        {
            if (directoryPath == null)
                throw new ArgumentNullException(nameof(directoryPath));

            if (!Directory.Exists(directoryPath))
                throw new DirectoryNotFoundException(directoryPath);

            var configPath = Path.Combine(directoryPath, ModernModelDirectoryWriter.ConfigFileName);
            var weightsPath = Path.Combine(directoryPath, ModernModelDirectoryWriter.WeightsFileName);
            var generationConfigPath = Path.Combine(directoryPath, ModernModelDirectoryWriter.GenerationConfigFileName);

            var config = ModelConfigJsonConverter.FromJson(File.ReadAllText(configPath));
            var weights = SafeTensorsReader.Read(weightsPath);

            var generationConfig = File.Exists(generationConfigPath)
                ? GenerationConfigJsonConverter.FromJson(File.ReadAllText(generationConfigPath))
                : GenerationConfigOptions.Default();

            return new ModernModelDirectoryContent(config, weights, generationConfig);
        }
    }
}
