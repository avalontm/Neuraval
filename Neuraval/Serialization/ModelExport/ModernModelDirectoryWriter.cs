using System;
using System.Collections.Generic;
using System.IO;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.SafeTensors;

namespace Neuraval.Core.Serialization.ModelExport
{
    public static class ModernModelDirectoryWriter
    {
        public const string ConfigFileName = "config.json";
        public const string WeightsFileName = "model.safetensors";
        public const string GenerationConfigFileName = "generation_config.json";

        public static void Write(
            string directoryPath,
            TransformerConfig config,
            IReadOnlyList<SafeTensorsEntry> tensors,
            GenerationConfigOptions? generationConfig = null)
        {
            if (directoryPath == null)
                throw new ArgumentNullException(nameof(directoryPath));

            if (config == null)
                throw new ArgumentNullException(nameof(config));

            if (tensors == null)
                throw new ArgumentNullException(nameof(tensors));

            config.Validate();

            if (!Directory.Exists(directoryPath))
            {
                Directory.CreateDirectory(directoryPath);
            }

            File.WriteAllText(Path.Combine(directoryPath, ConfigFileName), ModelConfigJsonConverter.ToJson(config));

            SafeTensorsWriter.Write(Path.Combine(directoryPath, WeightsFileName), tensors);

            var effectiveGenerationConfig = generationConfig ?? GenerationConfigOptions.Default();
            File.WriteAllText(
                Path.Combine(directoryPath, GenerationConfigFileName),
                GenerationConfigJsonConverter.ToJson(effectiveGenerationConfig));
        }
    }
}
