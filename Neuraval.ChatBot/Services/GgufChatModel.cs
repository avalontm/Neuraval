using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Neuraval.Abstractions;
using Neuraval.Core.Generation;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization;
using Neuraval.Core.Serialization.Gguf;
using Neuraval.Core.Tokenizers;

namespace Neuraval.ChatBot.Services
{
    public sealed class GgufChatModelOptions
    {
        public bool Greedy { get; set; } = false;
        public float Temperature { get; set; } = 0.2f;
        public float TopP { get; set; } = 0.9f;
        public int TopK { get; set; } = 40;
        public float RepetitionPenalty { get; set; } = 1.0f;
        public int MaxNewTokens { get; set; } = 256;
        public int MaxContextTokens { get; set; } = 2048;
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

        public GenerationPerformanceReport? LastPerformanceReport { get; private set; }

        public GgufChatModel(ModernDecoderModel model, IChatTokenizer tokenizer, GgufChatModelOptions? options = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            _options = options ?? new GgufChatModelOptions();
            if (_options.MaxContextTokens < 2)
                throw new ArgumentOutOfRangeException(nameof(options), "MaxContextTokens debe ser al menos 2.");
            _generator = new TextGenerator(_model);
            _stopTokenIds = BuildStopTokenIds(_tokenizer);
            _stopTokenIdSet = new HashSet<int>(_stopTokenIds);
        }

        public static GgufChatModel FromFile(string filePath, GgufChatModelOptions? options = null)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("filePath no puede ser vacío", nameof(filePath));

            if (string.Equals(System.IO.Path.GetExtension(filePath), ModernDecoderBinarySerializer.FileExtension, StringComparison.OrdinalIgnoreCase))
            {
                var nativeFile = ModernDecoderBinarySerializer.Load(filePath);
                options ??= new GgufChatModelOptions();
                options.ChatTemplate ??= nativeFile.ChatTemplate;
                return new GgufChatModel(nativeFile.Model, nativeFile.Tokenizer, options);
            }

            var file = GgufReader.Read(filePath);
            var model = GgufModelWeightLoader.Load(file, inferenceOnly: true);
            var tokenizer = GgufTokenizerLoader.Load(file);

            var architecture = GgufModelLoader.LoadConfig(file).Architecture;
            options ??= new GgufChatModelOptions();
            options.ChatTemplate ??= ResolveChatTemplate(file, tokenizer, architecture);

            return new GgufChatModel(model, tokenizer, options);
        }

        public static void ConvertToNativeFile(string ggufPath, string outputPath)
        {
            if (string.IsNullOrWhiteSpace(ggufPath))
                throw new ArgumentException("ggufPath no puede ser vacío", nameof(ggufPath));
            if (string.IsNullOrWhiteSpace(outputPath))
                throw new ArgumentException("outputPath no puede ser vacío", nameof(outputPath));

            var payload = ReadNativeConversionPayload(ggufPath);
            // The GGUF reader expands quantized tensors into float arrays. Release those
            // source buffers before writing the converted weights to keep peak RAM lower.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            ModernDecoderBinarySerializer.Save(outputPath, payload.State, payload.Tokenizer, payload.Template);
        }

        private static NativeConversionPayload ReadNativeConversionPayload(string ggufPath)
        {
            var file = GgufReader.Read(ggufPath);
            // GgufReader no longer needs the original file byte buffer after parsing.
            // Collect it before duplicating decoded weights into the native state.
            GC.Collect();
            var config = GgufModelLoader.LoadConfig(file);
            var weights = GgufModelLoader.LoadWeights(file);
            var report = GgufModelLoader.ValidateWeights(config, weights);
            if (!report.IsComplete)
                throw new InvalidOperationException(
                    $"El GGUF no contiene todos los tensores requeridos. Faltan: {string.Join(", ", report.MissingNames)}");

            var state = GgufModelWeightLoader.BuildState(config, weights);
            var tokenizer = GgufTokenizerLoader.Load(file);
            var template = ResolveChatTemplate(file, tokenizer, config.Architecture);
            return new NativeConversionPayload(state, tokenizer, template);
        }

        private sealed record NativeConversionPayload(ModernDecoderModelState State, IChatTokenizer Tokenizer, ChatTemplateDefinition Template);

        private static ChatTemplateDefinition ResolveChatTemplate(GgufFile file, IChatTokenizer tokenizer, string architecture)
        {
            if (TryGetRawChatTemplate(file, out var rawTemplate) && !string.IsNullOrWhiteSpace(rawTemplate))
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

        private static bool TryGetRawChatTemplate(GgufFile file, out string rawTemplate)
        {
            if (file.Metadata.TryGetString("tokenizer.chat_template", out rawTemplate)
                && !string.IsNullOrWhiteSpace(rawTemplate))
                return true;

            if (file.Metadata.TryGetValue("tokenizer.chat_template", out var value)
                && value.TryGetArray(out var templates))
            {
                foreach (var template in templates)
                {
                    if (template.TryGetString(out rawTemplate) && !string.IsNullOrWhiteSpace(rawTemplate))
                        return true;
                }
            }

            rawTemplate = string.Empty;
            return false;
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
            return SendCoreAsync(messages, null, cancellationToken);
        }

        public Task<ChatMessage> SendStreamingAsync(
            IReadOnlyList<ChatMessage> messages,
            Action<string> onTextGenerated,
            CancellationToken cancellationToken = default)
        {
            if (onTextGenerated == null)
                throw new ArgumentNullException(nameof(onTextGenerated));

            return SendCoreAsync(messages, onTextGenerated, cancellationToken);
        }

        private Task<ChatMessage> SendCoreAsync(
            IReadOnlyList<ChatMessage> messages,
            Action<string>? onTextGenerated,
            CancellationToken cancellationToken)
        {
            if (messages == null)
                throw new ArgumentNullException(nameof(messages));

            if (messages.Count == 0)
                throw new ChatModelException("La conversación no contiene mensajes.");

            cancellationToken.ThrowIfCancellationRequested();

            return Task.Run(() =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var promptTokenIds = EncodeBoundedChat(messages);

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
                var streamedTokenIds = onTextGenerated == null ? null : new List<int>();
                int emittedCharacters = 0;
                string emittedText = string.Empty;
                Action<int>? tokenCallback = null;
                if (onTextGenerated != null)
                {
                    tokenCallback = tokenId =>
                    {
                        streamedTokenIds!.Add(tokenId);
                        string decoded = _tokenizer.Decode(streamedTokenIds.ToArray(), skipSpecialTokens: true);
                        int stableLength = decoded.Length;
                        while (stableLength > 0 && decoded[stableLength - 1] == '\uFFFD')
                            stableLength--;

                        if (stableLength > emittedCharacters && decoded.StartsWith(emittedText, StringComparison.Ordinal))
                        {
                            string chunk = decoded.Substring(emittedCharacters, stableLength - emittedCharacters);
                            emittedCharacters = stableLength;
                            emittedText += chunk;
                            onTextGenerated(chunk);
                        }
                    };
                }
                try
                {
                    output = _generator.GenerateWithCache(
                        promptTokenIds,
                        generationOptions,
                        onTokenGenerated: tokenCallback,
                        contextLimit: _options.MaxContextTokens);
                    LastPerformanceReport = output.Performance;
                }
                catch (Exception ex)
                {
                    throw new ChatModelException("Falló la generación con el modelo GGUF cargado in-process.", ex);
                }

                var generatedTokenIds = output.Result.GeneratedTokenIds
                    .Where(id => !_stopTokenIdSet.Contains(id))
                    .ToArray();

                var responseText = _tokenizer.Decode(generatedTokenIds, skipSpecialTokens: true);

                if (onTextGenerated != null && responseText.StartsWith(emittedText, StringComparison.Ordinal)
                    && responseText.Length > emittedCharacters)
                    onTextGenerated(responseText.Substring(emittedCharacters));

                return new ChatMessage(ChatRole.Assistant, responseText.Trim());
            }, cancellationToken);
        }

        private int[] EncodeBoundedChat(IReadOnlyList<ChatMessage> messages)
        {
            var systemMessages = messages.Where(message => message.Role == ChatRole.System).ToList();
            var recentMessages = messages.Where(message => message.Role != ChatRole.System).TakeLast(12).ToList();
            var boundedMessages = new List<ChatMessage>(systemMessages.Count + recentMessages.Count);
            boundedMessages.AddRange(systemMessages);
            boundedMessages.AddRange(recentMessages);

            var tokenIds = _tokenizer.EncodeChat(boundedMessages, _options.ChatTemplate, addGenerationPrompt: true);
            var contextLimit = Math.Min(_options.MaxContextTokens, _model.MaxPositionEmbeddings);

            while (tokenIds.Length > contextLimit)
            {
                var firstConversationMessage = boundedMessages.FindIndex(message => message.Role != ChatRole.System);
                var lastConversationMessage = boundedMessages.FindLastIndex(message => message.Role != ChatRole.System);
                if (firstConversationMessage < 0 || firstConversationMessage >= lastConversationMessage)
                {
                    break;
                }

                boundedMessages.RemoveAt(firstConversationMessage);
                tokenIds = _tokenizer.EncodeChat(boundedMessages, _options.ChatTemplate, addGenerationPrompt: true);
            }

            return tokenIds;
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
