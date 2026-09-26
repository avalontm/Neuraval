using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Neuraval.Abstractions;

namespace Neuraval.ChatBot.Services
{
    public class OpenAiCompatibleChatModel : IChatModel, IDisposable
    {
        private readonly HttpClient _httpClient;
        private readonly OpenAiCompatibleOptions _options;
        private readonly bool _ownsHttpClient;

        public OpenAiCompatibleChatModel(OpenAiCompatibleOptions options, HttpClient? httpClient = null)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _ownsHttpClient = httpClient is null;
            _httpClient = httpClient ?? new HttpClient();
            _httpClient.Timeout = _options.Timeout;

            if (!string.IsNullOrWhiteSpace(_options.ApiKey))
            {
                _httpClient.DefaultRequestHeaders.Authorization =
                    new AuthenticationHeaderValue("Bearer", _options.ApiKey);
            }
        }

        public async Task<ChatMessage> SendAsync(IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
        {
            if (messages is null || messages.Count == 0)
                throw new ChatModelException("The conversation does not contain any messages.");

            var request = new OpenAiChatCompletionRequest
            {
                Model = _options.Model,
                Messages = messages.Select(ToOpenAiMessage).ToArray(),
                Temperature = _options.Temperature,
                MaxTokens = _options.MaxTokens,
                Stream = false
            };

            var json = JsonSerializer.Serialize(request);
            using var content = new StringContent(json, Encoding.UTF8, "application/json");

            HttpResponseMessage response;
            try
            {
                response = await _httpClient
                    .PostAsync($"{_options.BaseUrl}/chat/completions", content, cancellationToken)
                    .ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is HttpRequestException || ex is TaskCanceledException)
            {
                throw new ChatModelException($"Failed to reach the OpenAI-compatible endpoint at {_options.BaseUrl}.", ex);
            }

            var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
                throw new ChatModelException($"OpenAI-compatible endpoint returned {(int)response.StatusCode} {response.ReasonPhrase}: {body}");

            OpenAiChatCompletionResponse? parsed;
            try
            {
                parsed = JsonSerializer.Deserialize<OpenAiChatCompletionResponse>(body);
            }
            catch (JsonException ex)
            {
                throw new ChatModelException("Failed to parse the OpenAI-compatible response.", ex);
            }

            var choice = parsed?.Choices?.FirstOrDefault();
            if (choice?.Message?.Content is null)
                throw new ChatModelException("The OpenAI-compatible response did not contain a message.");

            return new ChatMessage(ParseRole(choice.Message.Role), choice.Message.Content);
        }

        private static OpenAiChatMessageDto ToOpenAiMessage(ChatMessage message)
        {
            return new OpenAiChatMessageDto
            {
                Role = ToRoleString(message.Role),
                Content = message.Content
            };
        }

        private static string ToRoleString(ChatRole role)
        {
            switch (role)
            {
                case ChatRole.System:
                    return "system";
                case ChatRole.User:
                    return "user";
                case ChatRole.Assistant:
                    return "assistant";
                case ChatRole.Tool:
                    return "tool";
                default:
                    throw new ChatModelException($"Unsupported chat role: {role}");
            }
        }

        private static ChatRole ParseRole(string? role)
        {
            switch (role)
            {
                case "system":
                    return ChatRole.System;
                case "user":
                    return ChatRole.User;
                case "tool":
                    return ChatRole.Tool;
                default:
                    return ChatRole.Assistant;
            }
        }

        public void Dispose()
        {
            if (_ownsHttpClient)
                _httpClient.Dispose();
        }
    }
}
