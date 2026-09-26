using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Neuraval.Abstractions;
using Neuraval.ChatBot.Services;
using Neuraval.Core.Models;
using Neuraval.Core.Tokenizers;
using Xunit;

namespace Neuraval.Tests
{
    public class GgufChatModelTests
    {
        private static (ModernDecoderModel Model, ModernBpeTokenizer Tokenizer) BuildTinyModelAndTokenizer()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.Train(new List<string> { "hola mundo", "chau mundo", "hola de nuevo" }, numMerges: 10, minPairFrequency: 1);

            var config = new TransformerConfig
            {
                VocabSize = tokenizer.VocabSize,
                HiddenSize = 8,
                NumHiddenLayers = 1,
                NumAttentionHeads = 2,
                NumKeyValueHeads = 2,
                IntermediateSize = 16,
                MaxPositionEmbeddings = 32,
                TieWordEmbeddings = true
            };

            var model = new ModernDecoderModel(config, seed: 7);

            return (model, tokenizer);
        }

        [Fact]
        public async Task SendAsync_GreedyDecoding_ReturnsAssistantMessage()
        {
            var (model, tokenizer) = BuildTinyModelAndTokenizer();
            var chatModel = new GgufChatModel(model, tokenizer, new GgufChatModelOptions { Greedy = true, MaxNewTokens = 3 });

            var response = await chatModel.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hola") });

            Assert.Equal(ChatRole.Assistant, response.Role);
            Assert.NotNull(response.Content);
        }

        [Fact]
        public async Task SendAsync_SamplingDecoding_ReturnsAssistantMessage()
        {
            var (model, tokenizer) = BuildTinyModelAndTokenizer();
            var options = new GgufChatModelOptions
            {
                Greedy = false,
                Temperature = 0.8f,
                TopP = 0.95f,
                TopK = 10,
                RepetitionPenalty = 1.1f,
                MaxNewTokens = 3,
                Seed = 123
            };
            var chatModel = new GgufChatModel(model, tokenizer, options);

            var response = await chatModel.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hola") });

            Assert.Equal(ChatRole.Assistant, response.Role);
            Assert.NotNull(response.Content);
        }

        [Fact]
        public async Task SendAsync_EmptyMessages_ThrowsChatModelException()
        {
            var (model, tokenizer) = BuildTinyModelAndTokenizer();
            var chatModel = new GgufChatModel(model, tokenizer);

            await Assert.ThrowsAsync<ChatModelException>(() => chatModel.SendAsync(Array.Empty<ChatMessage>()));
        }

        [Fact]
        public async Task SendAsync_AlreadyCancelledToken_ThrowsOperationCanceledException()
        {
            var (model, tokenizer) = BuildTinyModelAndTokenizer();
            var chatModel = new GgufChatModel(model, tokenizer);

            using var cts = new CancellationTokenSource();
            cts.Cancel();

            await Assert.ThrowsAsync<OperationCanceledException>(
                () => chatModel.SendAsync(new List<ChatMessage> { new(ChatRole.User, "hola") }, cts.Token));
        }

        [Fact]
        public void Constructor_NullModel_ThrowsArgumentNullException()
        {
            var (_, tokenizer) = BuildTinyModelAndTokenizer();

            Assert.Throws<ArgumentNullException>(() => new GgufChatModel(null!, tokenizer));
        }

        [Fact]
        public void Constructor_NullTokenizer_ThrowsArgumentNullException()
        {
            var (model, _) = BuildTinyModelAndTokenizer();

            Assert.Throws<ArgumentNullException>(() => new GgufChatModel(model, null!));
        }

        [Fact]
        public void FromFile_MissingFile_ThrowsFileNotFoundException()
        {
            var missingPath = Path.Combine(Path.GetTempPath(), $"neuraval-nonexistent-{Guid.NewGuid():N}.gguf");

            Assert.Throws<FileNotFoundException>(() => GgufChatModel.FromFile(missingPath));
        }
    }
}
