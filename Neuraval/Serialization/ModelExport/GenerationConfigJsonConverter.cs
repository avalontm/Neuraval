using System;
using System.Text.Json;

namespace Neuraval.Core.Serialization.ModelExport
{
    public static class GenerationConfigJsonConverter
    {
        private static readonly JsonSerializerOptions SerializerOptions = new()
        {
            WriteIndented = true
        };

        public static string ToJson(GenerationConfigOptions options)
        {
            if (options == null)
                throw new ArgumentNullException(nameof(options));

            var document = new GenerationConfigDocument
            {
                Temperature = options.Temperature,
                TopP = options.TopP,
                TopK = options.TopK,
                MaxNewTokens = options.MaxNewTokens
            };

            return JsonSerializer.Serialize(document, SerializerOptions);
        }

        public static GenerationConfigOptions FromJson(string json)
        {
            if (json == null)
                throw new ArgumentNullException(nameof(json));

            var document = JsonSerializer.Deserialize<GenerationConfigDocument>(json)
                ?? throw new InvalidOperationException("No se pudo deserializar generation_config.json");

            return GenerationConfigOptions.Create(document.Temperature, document.TopP, document.TopK, document.MaxNewTokens);
        }
    }
}
