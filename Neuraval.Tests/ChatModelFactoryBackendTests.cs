using Neuraval.Abstractions;
using Neuraval.ChatBot.Services;
using Xunit;

namespace Neuraval.Tests
{
    public class ChatModelFactoryBackendTests
    {
        [Fact]
        public void CreateFromConfigJson_ReturnsOpenAiCompatibleChatModel_ForOpenAiCompatibleBackend()
        {
            var json = "{\"backend\":\"openai-compatible\",\"base_url\":\"http://localhost:8080/v1\",\"model\":\"qwen\",\"api_key\":\"\"}";

            IChatModel model = ChatModelFactory.CreateFromConfigJson(json);

            Assert.IsType<OpenAiCompatibleChatModel>(model);
        }

        [Fact]
        public void CreateFromConfigJson_ReturnsLlamaCppChatModel_ForLlamaCppBackend()
        {
            var json = "{\"backend\":\"llama.cpp\",\"base_url\":\"http://localhost:8080/v1\",\"model\":\"qwen\"}";

            IChatModel model = ChatModelFactory.CreateFromConfigJson(json);

            Assert.IsType<LlamaCppChatModel>(model);
        }

        [Fact]
        public void CreateFromConfigJson_UsesLlamaCppDefaults_WhenBaseUrlAndModelOmitted()
        {
            var json = "{\"backend\":\"llamacpp\"}";

            IChatModel model = ChatModelFactory.CreateFromConfigJson(json);

            Assert.IsType<LlamaCppChatModel>(model);
        }

        [Fact]
        public void CreateFromConfigJson_Throws_ForUnknownBackend()
        {
            var json = "{\"backend\":\"unknown-backend\",\"base_url\":\"http://x\",\"model\":\"m\"}";

            Assert.Throws<ChatModelException>(() => ChatModelFactory.CreateFromConfigJson(json));
        }

        [Fact]
        public void CreateFromConfigJson_Throws_OnInvalidJson()
        {
            Assert.Throws<ChatModelException>(() => ChatModelFactory.CreateFromConfigJson("not json"));
        }
    }
}
