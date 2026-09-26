using System;
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
    public class OpenAiCompatibleChatModelTests
    {
        private static OpenAiCompatibleChatModel BuildModel(FakeHttpMessageHandler handler, string apiKey = "")
        {
            var options = new OpenAiCompatibleOptions("http://localhost:8080/v1", "qwen", apiKey);
            return new OpenAiCompatibleChatModel(options, new HttpClient(handler));
        }

        [Fact]
        public async Task SendAsync_ReturnsAssistantMessage_OnSuccess()
        {
            var handler = new FakeHttpMessageHandler(_ => FakeHttpMessageHandler.JsonResponse(
                HttpStatusCode.OK,
                "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hola\"},\"finish_reason\":\"stop\"}]}"));

            var model = BuildModel(handler);
            var response = await model.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hi") });

            Assert.Equal(ChatRole.Assistant, response.Role);
            Assert.Equal("hola", response.Content);
        }

        [Fact]
        public async Task SendAsync_PostsToChatCompletionsEndpoint_WithModelAndMessages()
        {
            var handler = new FakeHttpMessageHandler(_ => FakeHttpMessageHandler.JsonResponse(
                HttpStatusCode.OK,
                "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}"));

            var model = BuildModel(handler);
            await model.SendAsync(new List<ChatMessage>
            {
                new(ChatRole.System, "eres util"),
                new(ChatRole.User, "hola")
            });

            Assert.Equal("http://localhost:8080/v1/chat/completions", handler.LastRequest!.RequestUri!.ToString());

            using var payload = JsonDocument.Parse(handler.LastRequestBody!);
            var root = payload.RootElement;

            Assert.Equal("qwen", root.GetProperty("model").GetString());
            Assert.Equal(2, root.GetProperty("messages").GetArrayLength());
            Assert.Equal("system", root.GetProperty("messages")[0].GetProperty("role").GetString());
            Assert.Equal("user", root.GetProperty("messages")[1].GetProperty("role").GetString());
        }

        [Fact]
        public async Task SendAsync_SetsAuthorizationHeader_WhenApiKeyProvided()
        {
            var handler = new FakeHttpMessageHandler(_ => FakeHttpMessageHandler.JsonResponse(
                HttpStatusCode.OK,
                "{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}"));

            var model = BuildModel(handler, apiKey: "secret");
            await model.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hi") });

            Assert.Equal("Bearer", handler.LastRequest!.Headers.Authorization!.Scheme);
            Assert.Equal("secret", handler.LastRequest!.Headers.Authorization!.Parameter);
        }

        [Fact]
        public async Task SendAsync_Throws_WhenMessagesEmpty()
        {
            var handler = new FakeHttpMessageHandler(_ => FakeHttpMessageHandler.JsonResponse(HttpStatusCode.OK, "{}"));
            var model = BuildModel(handler);

            await Assert.ThrowsAsync<ChatModelException>(() => model.SendAsync(new List<ChatMessage>()));
        }

        [Fact]
        public async Task SendAsync_Throws_OnNonSuccessStatusCode()
        {
            var handler = new FakeHttpMessageHandler(_ => FakeHttpMessageHandler.JsonResponse(
                HttpStatusCode.InternalServerError, "{\"error\":\"boom\"}"));

            var model = BuildModel(handler);

            var ex = await Assert.ThrowsAsync<ChatModelException>(
                () => model.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hi") }));

            Assert.Contains("500", ex.Message);
        }

        [Fact]
        public async Task SendAsync_Throws_OnMalformedJson()
        {
            var handler = new FakeHttpMessageHandler(_ => FakeHttpMessageHandler.JsonResponse(
                HttpStatusCode.OK, "not json"));

            var model = BuildModel(handler);

            await Assert.ThrowsAsync<ChatModelException>(
                () => model.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hi") }));
        }

        [Fact]
        public async Task SendAsync_Throws_WhenNoChoicesReturned()
        {
            var handler = new FakeHttpMessageHandler(_ => FakeHttpMessageHandler.JsonResponse(
                HttpStatusCode.OK, "{\"choices\":[]}"));

            var model = BuildModel(handler);

            await Assert.ThrowsAsync<ChatModelException>(
                () => model.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hi") }));
        }

        [Fact]
        public void Constructor_Throws_WhenBaseUrlMissing()
        {
            Assert.Throws<ArgumentException>(() => new OpenAiCompatibleOptions("", "qwen"));
        }

        [Fact]
        public void Constructor_Throws_WhenModelMissing()
        {
            Assert.Throws<ArgumentException>(() => new OpenAiCompatibleOptions("http://localhost:8080/v1", ""));
        }
    }
}
