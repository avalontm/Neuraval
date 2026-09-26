using System.Text.Json.Serialization;

namespace Neuraval.ChatBot.Services
{
    internal sealed class OpenAiChatCompletionRequest
    {
        [JsonPropertyName("model")]
        public string Model { get; set; } = "";

        [JsonPropertyName("messages")]
        public OpenAiChatMessageDto[] Messages { get; set; } = System.Array.Empty<OpenAiChatMessageDto>();

        [JsonPropertyName("temperature")]
        public double? Temperature { get; set; }

        [JsonPropertyName("max_tokens")]
        public int? MaxTokens { get; set; }

        [JsonPropertyName("stream")]
        public bool Stream { get; set; }
    }

    internal sealed class OpenAiChatMessageDto
    {
        [JsonPropertyName("role")]
        public string Role { get; set; } = "";

        [JsonPropertyName("content")]
        public string Content { get; set; } = "";
    }

    internal sealed class OpenAiChatCompletionResponse
    {
        [JsonPropertyName("choices")]
        public OpenAiChatCompletionChoice[]? Choices { get; set; }
    }

    internal sealed class OpenAiChatCompletionChoice
    {
        [JsonPropertyName("message")]
        public OpenAiChatMessageDto? Message { get; set; }

        [JsonPropertyName("finish_reason")]
        public string? FinishReason { get; set; }
    }
}
