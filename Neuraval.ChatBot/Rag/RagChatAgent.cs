using Neuraval.Abstractions;
using Neuraval.ChatBot.Services;
using Neuraval.Core.Generation;

namespace Neuraval.ChatBot.Rag
{
    /// <summary>Retrieves local document passages before delegating each turn to a chat model.</summary>
    public sealed class RagChatAgent : IChatModel, IDisposable
    {
        private readonly IChatModel _model;
        private readonly RagKnowledgeBase _knowledgeBase;
        private readonly int _topK;

        public bool SupportsStreaming => _model is GgufChatModel;
        public GenerationPerformanceReport? LastPerformanceReport => (_model as GgufChatModel)?.LastPerformanceReport;

        public RagChatAgent(IChatModel model, RagKnowledgeBase knowledgeBase, int topK = 3)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _knowledgeBase = knowledgeBase ?? throw new ArgumentNullException(nameof(knowledgeBase));
            if (topK <= 0) throw new ArgumentOutOfRangeException(nameof(topK));
            _topK = topK;
        }

        public IReadOnlyList<RagSearchResult> Retrieve(string query) => _knowledgeBase.Search(query, _topK);

        public Task<ChatMessage> SendAsync(IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
        {
            var context = BuildContextMessages(messages);
            return _model.SendAsync(context, cancellationToken);
        }

        public Task<ChatMessage> SendStreamingAsync(
            IReadOnlyList<ChatMessage> messages,
            Action<string> onTextGenerated,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(onTextGenerated);
            var context = BuildContextMessages(messages);
            if (_model is GgufChatModel gguf)
            {
                return gguf.SendStreamingAsync(context, onTextGenerated, cancellationToken);
            }

            return _model.SendAsync(context, cancellationToken);
        }

        private List<ChatMessage> BuildContextMessages(IReadOnlyList<ChatMessage> messages)
        {
            ArgumentNullException.ThrowIfNull(messages);
            var question = messages.LastOrDefault(message => message.Role == ChatRole.User)?.Content ?? string.Empty;
            var results = Retrieve(question);
            var system = new System.Text.StringBuilder();
            system.AppendLine("Eres un agente RAG. Responde usando primero la documentación local proporcionada.");
            system.AppendLine("Si el contexto no contiene la respuesta, dilo claramente y no inventes datos.");
            system.AppendLine("Cuando uses un fragmento, cita su archivo entre corchetes, por ejemplo [manual.md].");
            system.AppendLine("Contexto recuperado:");

            if (results.Count == 0)
            {
                system.AppendLine("No se encontraron fragmentos relevantes para esta pregunta.");
            }
            else
            {
                foreach (var result in results)
                {
                    system.Append('[').Append(result.Chunk.Source).AppendLine("]");
                    system.AppendLine(result.Chunk.Text);
                    system.AppendLine();
                }
            }

            var contextualMessages = new List<ChatMessage>(messages.Count + 1)
            {
                new(ChatRole.System, system.ToString())
            };
            contextualMessages.AddRange(messages);
            return contextualMessages;
        }

        public void Dispose()
        {
            if (_model is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }
}
