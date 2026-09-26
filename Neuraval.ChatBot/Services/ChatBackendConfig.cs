using System.Text.Json.Serialization;

namespace Neuraval.ChatBot.Services
{
    public class ChatBackendConfig
    {
        [JsonPropertyName("backend")]
        public string Backend { get; set; } = "";

        [JsonPropertyName("base_url")]
        public string BaseUrl { get; set; } = "";

        [JsonPropertyName("model")]
        public string Model { get; set; } = "";

        [JsonPropertyName("api_key")]
        public string ApiKey { get; set; } = "";

        [JsonPropertyName("temperature")]
        public double? Temperature { get; set; }

        [JsonPropertyName("max_tokens")]
        public int? MaxTokens { get; set; }
    }
}
