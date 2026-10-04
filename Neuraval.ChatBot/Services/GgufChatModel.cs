using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Neuraval.Abstractions;
using Neuraval.Core.Generation;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization.Gguf;
using Neuraval.Core.Tokenizers;

namespace Neuraval.ChatBot.Services
{
    public sealed class GgufChatModelOptions
    {
        public bool Greedy { get; set; } = true;
        public float Temperature { get; set; } = 0.7f;
        public float TopP { get; set; } = 0.9f;
        public int TopK { get; set; } = 40;
        public float RepetitionPenalty { get; set; } = 1.1f;
        public int MaxNewTokens { get; set; } = 256;
        public int? Seed { get; set; }
        public ChatTemplateDefinition? ChatTemplate { get; set; }
    }

    public sealed class GgufChatModel : IChatModel
    {
        private readonly ModernDecoderModel _model;
        private readonly IChatTokenizer _tokenizer;
        private readonly TextGenerator _generator;
        private readonly GgufChatModelOptions _options;
        private readonly int[] _stopTokenIds;
        private readonly HashSet<int> _stopTokenIdSet;

        public GgufChatModel(ModernDecoderModel model, IChatTokenizer tokenizer, GgufChatModelOptions? options = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _options = options ?? new GgufChatModelOptions();
            _generator = new TextGenerator(_model);
            _stopTokenIds = BuildStopTokenIds(_tokenizer);
            _stopTokenIdSet = new HashSet<int>(_stopTokenIds);
        }

        public static GgufChatModel FromFile(string filePath, GgufChatModelOptions? options = null)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("filePath no puede ser vacío", nameof(filePath));

            var file = GgufReader.Read(filePath);
            var model = GgufModelWeightLoader.Load(file);
            var tokenizer = GgufTokenizerLoader.Load(file);

            var architecture = GgufModelLoader.LoadConfig(file).Architecture;
            options ??= new GgufChatModelOptions();
            options.ChatTemplate ??= ResolveChatTemplate(file, tokenizer, architecture);

            return new GgufChatModel(model, tokenizer, options);
        }

        private static ChatTemplateDefinition ResolveChatTemplate(GgufFile file, IChatTokenizer tokenizer, string architecture)
        {
            if (file.Metadata.TryGetString("tokenizer.chat_template", out var rawTemplate) && !string.IsNullOrWhiteSpace(rawTemplate))
            {
                var bosText = tokenizer.GetToken(tokenizer.StartToken);
                var eosText = tokenizer.GetToken(tokenizer.EndToken);
                var candidate = ChatTemplateDefinition.FromRawJinja(rawTemplate, bosText, eosText);

                try
                {
                    var probeMessages = new List<ChatMessage> { new(ChatRole.User, "probe") };
                    ChatTemplateEngine.Render(probeMessages, candidate, addGenerationPrompt: true);
                    return candidate;
                }
                catch (JinjaTemplateException)
                {
                }
            }

            return ResolveDefaultChatTemplate(architecture);
        }

        private static ChatTemplateDefinition ResolveDefaultChatTemplate(string architecture)
        {
            return architecture.ToLowerInvariant() switch
            {
                "llama" or "mistral" => ChatTemplateDefinition.Mistral(),
                _ => ChatTemplateDefinition.ChatMl()
            };
        }

        public Task<ChatMessage> SendAsync(IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
        {
            if (messages == null)
                throw new ArgumentNullException(nameof(messages));

            if (messages.Count == 0)
                throw new ChatModelException("La conversación no contiene mensajes.");

            cancellationToken.ThrowIfCancellationRequested();

            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var promptTokenIds = _tokenizer.EncodeChat(messages, _options.ChatTemplate, addGenerationPrompt: true);

                if (promptTokenIds.Length == 0)
                    throw new ChatModelException("El prompt codificado quedó vacío.");

                var generationOptions = GenerationOptions.Create(
                    _options.Greedy,
                    _options.Temperature,
                    _options.TopP,
                    _options.TopK,
                    _options.RepetitionPenalty,
                    _options.MaxNewTokens,
                    _stopTokenIds,
                    _options.Seed);

                CachedGenerationOutput output;
                try
                {
                    output = _generator.GenerateWithCache(promptTokenIds, generationOptions);
                }
                catch (Exception ex)
                {
                    throw new ChatModelException("Falló la generación con el modelo GGUF cargado in-process.", ex);
                }

                var generatedTokenIds = output.Result.GeneratedTokenIds
                    .Where(id => !_stopTokenIdSet.Contains(id))
                    .ToArray();

                var responseText = _tokenizer.Decode(generatedTokenIds, skipSpecialTokens: true);

                return new ChatMessage(ChatRole.Assistant, responseText.Trim());
            }, cancellationToken);
        }

        private static int[] BuildStopTokenIds(IChatTokenizer tokenizer)
        {
            var ids = new HashSet<int> { tokenizer.EndToken, tokenizer.ImEndToken };

            foreach (var pair in tokenizer.GetVocabulary())
            {
                var token = pair.Key;
                bool looksLikeControlMarker =
                    (token.StartsWith("<|", StringComparison.Ordinal) && token.EndsWith("|>", StringComparison.Ordinal))
                    || (token.StartsWith("<", StringComparison.Ordinal) && token.EndsWith(">", StringComparison.Ordinal)
                        && !token.StartsWith("<0x", StringComparison.Ordinal));

                if (!looksLikeControlMarker)
                    continue;

                var lowered = token.ToLowerInvariant();
                if (lowered.Contains("eot") || lowered.Contains("end_of_turn") || lowered.Contains("end_of_text"))
                    ids.Add(pair.Value);
            }

            return ids.ToArray();
        }
    }
}
