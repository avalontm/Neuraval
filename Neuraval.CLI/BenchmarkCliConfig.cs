using System.Text.Json;
using System.Text.Json.Serialization;

namespace Neuraval.CLI
{
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
        [JsonPropertyName("name")]
        public string Name { get; set; } = "";

        [JsonPropertyName("backend")]
        public string Backend { get; set; } = "";

        [JsonPropertyName("base_url")]
        public string BaseUrl { get; set; } = "";

        [JsonPropertyName("model")]
        public string Model { get; set; } = "";

        [JsonPropertyName("api_key")]
        public string ApiKey { get; set; } = "";

        [JsonPropertyName("model_path")]
        public string? ModelPath { get; set; }

        [JsonPropertyName("parameter_count")]
        public long? ParameterCount { get; set; }

        [JsonPropertyName("model_size_bytes")]
        public long? ModelSizeBytes { get; set; }
    }
}
