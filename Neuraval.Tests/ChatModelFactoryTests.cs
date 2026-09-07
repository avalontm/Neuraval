using System;
using System.Collections.Generic;
using System.IO;
using Neuraval.Abstractions;
using Neuraval.ChatBot.Services;
using Neuraval.Core.Services;
using Neuraval.Core.Utils;
using Xunit;

namespace Neuraval.Tests
{
    public class ChatModelFactoryTests
    {
        [Fact]
        public void CreateFromSavedModel_ReturnsChatModel_WhenModelPathDoesNotExist()
        {
            Matematicas.SetNumThreads(1);
            var missingPath = Path.Combine(Path.GetTempPath(), "nchatbot_missing_" + Guid.NewGuid());

            IChatModel chatModel = ChatModelFactory.CreateFromSavedModel(
                missingPath,
                embeddingDim: 8,
                numLayers: 1,
                numHeads: 1,
                feedforwardDim: 16,
                maxSequenceLength: 8,
                numThreads: 1);

            Assert.IsType<TransformerChatBotService>(chatModel);
        }

        [Fact]
        public void CreateFromSavedModel_LoadsPersistedModel_MatchesDirectLoad()
        {
            Matematicas.SetNumThreads(1);
            var modelFolder = Path.Combine(Path.GetTempPath(), "nchatbot_factory_" + Guid.NewGuid());
            Directory.CreateDirectory(modelFolder);

            try
            {
                var trainingService = new TransformerChatBotService(
                    embeddingDim: 8,
                    numLayers: 1,
                    numHeads: 1,
                    feedforwardDim: 16,
                    maxSequenceLength: 8,
                    numThreads: 1);

                trainingService.BuildVocabularyFromTexts(new List<string> { "hola", "como estas", "bien gracias" });

                var conversations = new List<ConversationPair>
                {
                    new ConversationPair { Input = "hola", Target = "bien gracias" }
                };

                trainingService.TrainWithConversations(conversations, epochs: 1, batchSize: 1, validationSplit: 0);
                trainingService.SaveCompleteModel(modelFolder);

                var directLoadService = new TransformerChatBotService(numThreads: 1);
                directLoadService.LoadCompleteModel(modelFolder);
                var expectedResponse = directLoadService.GenerateResponse("hola");

                IChatModel factoryModel = ChatModelFactory.CreateFromSavedModel(modelFolder, numThreads: 1);
                var factoryService = Assert.IsType<TransformerChatBotService>(factoryModel);
                var actualResponse = factoryService.GenerateResponse("hola");

                Assert.Equal(expectedResponse, actualResponse);
                Assert.True(factoryService.IsTrained());
            }
            finally
            {
                Directory.Delete(modelFolder, recursive: true);
            }
        }
    }
}
