using System;
using System.IO;
using System.Text.Json;
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

        public static IChatModel CreateFromConfigJson(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                throw new ChatModelException("The chat backend configuration JSON is empty.");
            }

            ChatBackendConfig? config;
            try
            {
                config = JsonSerializer.Deserialize<ChatBackendConfig>(json);
            }
            catch (JsonException ex)
            {
                throw new ChatModelException("The chat backend configuration is not valid JSON.", ex);
            }

            if (config == null)
            {
                throw new ChatModelException("The chat backend configuration is empty.");
            }

            return CreateFromBackendConfig(config);
        }

        public static IChatModel CreateFromBackendConfig(ChatBackendConfig config)
        {
            var backend = config.Backend?.Trim() ?? string.Empty;

            if (string.Equals(backend, "openai-compatible", StringComparison.OrdinalIgnoreCase))
            {
                if (string.IsNullOrWhiteSpace(config.BaseUrl) || string.IsNullOrWhiteSpace(config.Model))
                {
                    throw new ChatModelException(
                        "The 'openai-compatible' backend requires both 'base_url' and 'model'.");
                }

                var options = new OpenAiCompatibleOptions(config.BaseUrl, config.Model, config.ApiKey ?? string.Empty)
                {
                    Temperature = config.Temperature,
                    MaxTokens = config.MaxTokens
                };

                return new OpenAiCompatibleChatModel(options);
            }

            if (string.Equals(backend, "llama.cpp", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(backend, "llamacpp", StringComparison.OrdinalIgnoreCase))
            {
                var options = new LlamaCppOptions();

                if (!string.IsNullOrWhiteSpace(config.BaseUrl))
                {
                    options.BaseUrl = config.BaseUrl;
                }

                if (!string.IsNullOrWhiteSpace(config.Model))
                {
                    options.Model = config.Model;
                }

                options.ApiKey = config.ApiKey ?? string.Empty;
                options.Temperature = config.Temperature;
                options.MaxTokens = config.MaxTokens;

                return new LlamaCppChatModel(options);
            }

            throw new ChatModelException(
                $"Unknown chat backend '{config.Backend}'. Supported values: 'openai-compatible', 'llama.cpp', 'llamacpp'.");
        }
    }
}
