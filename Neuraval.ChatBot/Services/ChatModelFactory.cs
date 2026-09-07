using System.IO;
using Neuraval.Abstractions;

namespace Neuraval.ChatBot.Services
{
    public static class ChatModelFactory
    {
        public static IChatModel CreateFromSavedModel(
            string modelPath,
            int embeddingDim = 128,
            int numLayers = 4,
            int numHeads = 4,
            int feedforwardDim = 512,
            int maxSequenceLength = 128,
            double dropout = 0.1,
            int numThreads = -1)
        {
            var chatModel = new TransformerChatBotService(
                embeddingDim: embeddingDim,
                numLayers: numLayers,
                numHeads: numHeads,
                feedforwardDim: feedforwardDim,
                maxSequenceLength: maxSequenceLength,
                dropout: dropout,
                numThreads: numThreads);

            if (Directory.Exists(modelPath))
            {
                chatModel.LoadCompleteModel(modelPath);
            }

            return chatModel;
        }
    }
}
