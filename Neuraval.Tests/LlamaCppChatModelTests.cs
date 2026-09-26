using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Neuraval.Abstractions;
using Neuraval.ChatBot.Services;
using Xunit;

namespace Neuraval.Tests
{
    public class LlamaCppChatModelTests
    {
        [Fact]
        public async Task SendAsync_UsesDefaultLocalEndpoint_WhenOptionsOmitted()
        {
            var handler = new FakeHttpMessageHandler(_ => FakeHttpMessageHandler.JsonResponse(
                HttpStatusCode.OK,
                "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hola\"}}]}"));

            var model = new LlamaCppChatModel(httpClient: new HttpClient(handler));
            var response = await model.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hi") });

            Assert.Equal("hola", response.Content);
            Assert.Equal("http://localhost:8080/v1/chat/completions", handler.LastRequest!.RequestUri!.ToString());

            using var payload = JsonDocument.Parse(handler.LastRequestBody!);
            Assert.Equal("default", payload.RootElement.GetProperty("model").GetString());
        }

        [Fact]
        public async Task SendAsync_UsesConfiguredBaseUrlAndModel()
        {
            var handler = new FakeHttpMessageHandler(_ => FakeHttpMessageHandler.JsonResponse(
                HttpStatusCode.OK,
                "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}"));

            var options = new LlamaCppOptions(baseUrl: "http://192.168.1.10:8080/v1", model: "qwen2.5");
            var model = new LlamaCppChatModel(options, new HttpClient(handler));

            await model.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hi") });

            Assert.Equal("http://192.168.1.10:8080/v1/chat/completions", handler.LastRequest!.RequestUri!.ToString());

            using var payload = JsonDocument.Parse(handler.LastRequestBody!);
            Assert.Equal("qwen2.5", payload.RootElement.GetProperty("model").GetString());
        }
    }
}
