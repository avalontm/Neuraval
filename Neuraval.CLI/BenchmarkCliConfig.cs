using System.Text.Json;
using System.Text.Json.Serialization;

namespace Neuraval.CLI
{
    /// <summary>
    /// Config para "dotnet run --project Neuraval.CLI -- --chat-benchmark benchmark.json".
    /// Cada "subject" es un backend a comparar (Fase 25 del roadmap: Neuraval Native,
    /// Qwen/Gemma/Llama servidos por llama.cpp, GPT/API vía openai-compatible).
    /// </summary>
    public class BenchmarkCliConfig
    {
        [JsonPropertyName("prompts")]
        public List<string> Prompts { get; set; } = new();

        [JsonPropertyName("subjects")]
        public List<BenchmarkSubjectCliConfig> Subjects { get; set; } = new();

        public static BenchmarkCliConfig LoadFromFile(string filepath)
        {
            var json = File.ReadAllText(filepath);
            var config = JsonSerializer.Deserialize<BenchmarkCliConfig>(json);
            return config ?? new BenchmarkCliConfig();
        }
    }

    public class BenchmarkSubjectCliConfig
    {
        /// <summary>Nombre para mostrar en la tabla comparativa (ej. "Qwen 2.5 7B").</summary>
        [JsonPropertyName("name")]
        public string Name { get; set; } = "";

        /// <summary>"native" (modelo Neuraval entrenado localmente), o cualquier backend soportado
        /// por <see cref="Neuraval.ChatBot.Services.ChatModelFactory.CreateFromConfig"/>
        /// ("openai-compatible", "llama.cpp", etc.).</summary>
        [JsonPropertyName("backend")]
        public string Backend { get; set; } = "";

        [JsonPropertyName("base_url")]
        public string BaseUrl { get; set; } = "";

        [JsonPropertyName("model")]
        public string Model { get; set; } = "";

        [JsonPropertyName("api_key")]
        public string ApiKey { get; set; } = "";

        /// <summary>Solo para backend "native": carpeta con model.navm y tokenizer.json.</summary>
        [JsonPropertyName("model_path")]
        public string? ModelPath { get; set; }

        [JsonPropertyName("parameter_count")]
        public long? ParameterCount { get; set; }

        [JsonPropertyName("model_size_bytes")]
        public long? ModelSizeBytes { get; set; }
    }
}
