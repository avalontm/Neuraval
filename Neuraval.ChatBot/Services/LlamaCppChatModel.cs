using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Neuraval.Abstractions;

namespace Neuraval.ChatBot.Services
{
    public class LlamaCppChatModel : IChatModel, IDisposable
    {
        private readonly OpenAiCompatibleChatModel _inner;

        public LlamaCppChatModel(LlamaCppOptions? options = null, HttpClient? httpClient = null)
        {
            options ??= new LlamaCppOptions();

            var innerOptions = new OpenAiCompatibleOptions(options.BaseUrl, options.Model, options.ApiKey)
            {
                Temperature = options.Temperature,
                MaxTokens = options.MaxTokens
            };

            _inner = new OpenAiCompatibleChatModel(innerOptions, httpClient);
        }

        public Task<ChatMessage> SendAsync(IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
        {
            return _inner.SendAsync(messages, cancellationToken);
        }

        public void Dispose()
        {
            _inner.Dispose();
        }
    }
}
