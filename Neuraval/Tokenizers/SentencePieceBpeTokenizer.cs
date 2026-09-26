using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Neuraval.Abstractions;
using Neuraval.Core.Services;

namespace Neuraval.Core.Tokenizers
{
    // Reconstruye un tokenizer SentencePiece BPE (el usado por Llama/Mistral, declarado en GGUF
    // como tokenizer.ggml.model = "llama") a partir de vocabulario + scores, sin depender de una
    // lista explícita de merges: a diferencia de la BPE byte-level de ModernBpeTokenizer (donde
    // la prioridad de cada par la da su posición en `merges`), acá la prioridad la da el score de
    // cada token del vocabulario (`tokenizer.ggml.scores`), y en cada paso se fusiona el par
    // adyacente cuya concatenación exista en el vocabulario con el score más alto.
    public sealed class SentencePieceBpeTokenizer : IChatTokenizer
    {
        private const string SpaceSymbol = "\u2581"; // "▁", usado por SentencePiece en vez de ' '

        private Vocabulary _vocabulary = new();
        private Dictionary<int, float> _scores = new();
        private Dictionary<int, byte> _tokenTypes = new();
        private Dictionary<byte, int> _byteTokenId = new();
        private bool _addDummyPrefix = true;

        public int VocabSize => _vocabulary.Count;
        public int PadToken { get; private set; }
        public int UnknownToken { get; private set; }
        public int StartToken { get; private set; }
        public int EndToken { get; private set; }
        public int SepToken { get; private set; }
        public int ImStartToken { get; private set; }
        public int ImEndToken { get; private set; }

        public void BuildVocabulary(List<string> texts)
        {
            throw new NotSupportedException(
                "SentencePieceBpeTokenizer solo reconstruye tokenizers ya entrenados (desde GGUF); no soporta entrenamiento propio.");
        }

        public void BuildVocabulary(List<string> texts, int vocabSize)
        {
            throw new NotSupportedException(
                "SentencePieceBpeTokenizer solo reconstruye tokenizers ya entrenados (desde GGUF); no soporta entrenamiento propio.");
        }

        private string Normalize(string text, bool applyDummyPrefix)
        {
            var replaced = text.Replace(' ', SpaceSymbol[0]);

            if (applyDummyPrefix && _addDummyPrefix && !replaced.StartsWith(SpaceSymbol, StringComparison.Ordinal))
                replaced = SpaceSymbol + replaced;

            return replaced;
        }

        private static List<string> SplitIntoSymbols(string text)
        {
            var symbols = new List<string>();

            foreach (var rune in text.EnumerateRunes())
                symbols.Add(rune.ToString());

            return symbols;
        }

        private List<string> MergeSymbols(List<string> symbols)
        {
            while (symbols.Count > 1)
            {
                int bestIndex = -1;
                float bestScore = float.NegativeInfinity;

                for (int i = 0; i < symbols.Count - 1; i++)
                {
                    var merged = symbols[i] + symbols[i + 1];

                    if (!_vocabulary.TryGetId(merged, out var id) || !_scores.TryGetValue(id, out var score))
                        continue;

                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestIndex = i;
                    }
                }

                if (bestIndex < 0)
                    break;

                var next = new List<string>(symbols.Count - 1);
                next.AddRange(symbols.Take(bestIndex));
                next.Add(symbols[bestIndex] + symbols[bestIndex + 1]);
                next.AddRange(symbols.Skip(bestIndex + 2));
                symbols = next;
            }

            return symbols;
        }

        private void AppendWithByteFallback(string symbol, List<int> ids)
        {
            var bytes = Encoding.UTF8.GetBytes(symbol);
            var byteIds = new List<int>(bytes.Length);
            bool anyByteFound = false;

            foreach (var b in bytes)
            {
                if (_byteTokenId.TryGetValue(b, out var byteId))
                {
                    byteIds.Add(byteId);
                    anyByteFound = true;
                }
                else
                {
                    byteIds.Add(UnknownToken);
                }
            }

            if (anyByteFound)
                ids.AddRange(byteIds);
            else
                ids.Add(UnknownToken);
        }

        private List<int> EncodeToIds(string text) => EncodeToIds(text, applyDummyPrefix: true);

        private List<int> EncodeToIds(string text, bool applyDummyPrefix)
        {
            if (string.IsNullOrEmpty(text))
                return new List<int>();

            var symbols = MergeSymbols(SplitIntoSymbols(Normalize(text, applyDummyPrefix)));
            var ids = new List<int>(symbols.Count);

            foreach (var symbol in symbols)
            {
                if (_vocabulary.TryGetId(symbol, out var id))
                    ids.Add(id);
                else
                    AppendWithByteFallback(symbol, ids);
            }

            return ids;
        }

        public int[] Encode(string text, bool addSpecialTokens = true)
        {
            var ids = new List<int>();

            if (addSpecialTokens)
                ids.Add(StartToken);

            ids.AddRange(EncodeToIds(text));

            if (addSpecialTokens)
                ids.Add(EndToken);

            return ids.ToArray();
        }

        public int[] EncodeChat(IReadOnlyList<ChatMessage> messages, ChatTemplateDefinition? template = null, bool addGenerationPrompt = true)
        {
            var rendered = ChatTemplateEngine.Render(messages, template, addGenerationPrompt);

            if (template?.RawJinjaTemplate != null)
            {
                // El chat_template real del GGUF decide por su cuenta dónde va el BOS
                // (normalmente vía "{{ bos_token }}"), así que acá no podemos anteponer
                // StartToken a ciegas como en el camino "legacy" de abajo: eso
                // duplicaría el BOS en modelos cuyo template ya lo incluye. En vez de
                // eso, tokenizamos reconociendo los tokens especiales literales (<s>,
                // </s>, <|im_start|>, etc.) -- un merge score-based normal casi nunca
                // reconstruye esos tokens de forma fiable a partir de texto plano -- y
                // solo agregamos BOS si no quedó ya como primer token.
                return EncodeRenderedChatText(rendered);
            }

            // BUGFIX: antes se devolvía EncodeToIds(rendered) directo, sin token de
            // inicio (BOS). Modelos tipo Llama/Mistral fueron entrenados para ver
            // siempre BOS como primer token de la secuencia; sin él, el primer
            // forward pass queda fuera de distribución y la generación degenera
            // en ruido (tokens random de vocabularios/idiomas sin relación).
            var ids = new List<int> { StartToken };
            ids.AddRange(EncodeToIds(rendered));
            return ids.ToArray();
        }

        private int[] EncodeRenderedChatText(string renderedText)
        {
            var specialTokens = BuildSpecialTokenLiterals();
            var ids = new List<int>();
            bool isFirstPlainSegment = true;

            foreach (var segment in SpecialTokenTextSplitter.Split(renderedText, specialTokens))
            {
                if (segment.IsSpecial)
                {
                    ids.Add(segment.TokenId);
                    continue;
                }

                // El prefijo de espacio de SentencePiece se aplica una sola vez al
                // principio de la secuencia real, no en cada fragmento de texto plano
                // que quedó separado por un token especial en el medio.
                ids.AddRange(EncodeToIds(segment.Text, applyDummyPrefix: isFirstPlainSegment));
                isFirstPlainSegment = false;
            }

            if (ids.Count == 0 || ids[0] != StartToken)
                ids.Insert(0, StartToken);

            return ids.ToArray();
        }

        private Dictionary<string, int> BuildSpecialTokenLiterals()
        {
            var map = new Dictionary<string, int>();

            void AddIfPresent(int id)
            {
                if (_vocabulary.IdToToken.TryGetValue(id, out var literal) && literal.Length > 0)
                    map[literal] = id;
            }

            AddIfPresent(StartToken);
            AddIfPresent(EndToken);
            AddIfPresent(SepToken);
            AddIfPresent(ImStartToken);
            AddIfPresent(ImEndToken);

            foreach (var pair in _tokenTypes)
            {
                if (pair.Value == 3)
                    AddIfPresent(pair.Key);
            }

            return map;
        }

        public int[] EncodeCausalSequence(string prompt, string response)
        {
            var ids = new List<int> { StartToken };
            ids.AddRange(EncodeToIds(prompt));
            ids.Add(SepToken);
            ids.AddRange(EncodeToIds(response));
            ids.Add(EndToken);

            return ids.ToArray();
        }

        public int[] EncodePrompt(string prompt)
        {
            var ids = new List<int> { StartToken };
            ids.AddRange(EncodeToIds(prompt));
            ids.Add(SepToken);

            return ids.ToArray();
        }

        private bool TryGetByteValue(string token, out byte value)
        {
            if (token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token.EndsWith(">", StringComparison.Ordinal)
                && byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out var parsed))
            {
                value = parsed;
                return true;
            }

            value = 0;
            return false;
        }

        private bool IsControlToken(int id)
        {
            return _tokenTypes.TryGetValue(id, out var type) && type == 3; // LLAMA_TOKEN_TYPE_CONTROL
        }

        private bool IsSpecialOrControlToken(int id)
        {
            // Antes acá solo se comparaba el texto del token contra los placeholders
            // internos de SpecialTokens ("<|bos|>", etc.), que para un vocabulario
            // cargado desde un GGUF real casi nunca coinciden con el literal propio
            // del modelo ("<s>", "<|im_end|>", ...). Se decide por ID en vez de por
            // texto para que esto funcione igual con vocabularios entrenados
            // internamente y con los que vienen de un GGUF.
            return id == PadToken || id == UnknownToken || id == StartToken || id == EndToken
                || id == SepToken || id == ImStartToken || id == ImEndToken
                || IsControlToken(id);
        }

        public string Decode(int[] tokenIds, bool skipSpecialTokens = true)
        {
            var output = new StringBuilder();
            var byteBuffer = new List<byte>();

            void FlushBytes()
            {
                if (byteBuffer.Count == 0)
                    return;

                output.Append(Encoding.UTF8.GetString(byteBuffer.ToArray()));
                byteBuffer.Clear();
            }

            foreach (var id in tokenIds)
            {
                var token = _vocabulary.GetToken(id, SpecialTokens.Unk);

                if (IsSpecialOrControlToken(id))
                {
                    FlushBytes();

                    if (!skipSpecialTokens)
                        output.Append(token);

                    continue;
                }

                if (TryGetByteValue(token, out var byteValue))
                {
                    byteBuffer.Add(byteValue);
                    continue;
                }

                FlushBytes();
                output.Append(token);
            }

            FlushBytes();

            var text = output.ToString().Replace(SpaceSymbol, " ");

            if (text.StartsWith(" ", StringComparison.Ordinal))
                text = text.Substring(1);

            return text;
        }

        public int[] PadSequence(int[] sequence, int maxLength, bool padLeft = false)
        {
            if (sequence.Length >= maxLength)
                return sequence.Take(maxLength).ToArray();

            var padded = new int[maxLength];

            for (int i = 0; i < maxLength; i++)
                padded[i] = PadToken;

            if (padLeft)
            {
                int offset = maxLength - sequence.Length;
                Array.Copy(sequence, 0, padded, offset, sequence.Length);
            }
            else
            {
                Array.Copy(sequence, 0, padded, 0, sequence.Length);
            }

            return padded;
        }

        public List<int[]> PadBatch(List<int[]> sequences, int? maxLength = null)
        {
            int batchMaxLength = maxLength ?? sequences.Max(s => s.Length);
            return sequences.Select(seq => PadSequence(seq, batchMaxLength)).ToList();
        }

        public bool ContainsToken(string token)
        {
            return _vocabulary.Contains(token);
        }

        public int GetTokenId(string token)
        {
            return _vocabulary.GetId(token, UnknownToken);
        }

        public string GetToken(int id)
        {
            return _vocabulary.GetToken(id, SpecialTokens.Unk);
        }

        public Dictionary<string, int> GetVocabulary()
        {
            return new Dictionary<string, int>(_vocabulary.TokenToId);
        }

        public void SaveToFile(string filepath)
        {
            var state = SaveState();
            var json = System.Text.Json.JsonSerializer.Serialize(state, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });
            System.IO.File.WriteAllText(filepath, json);
        }

        public SentencePieceBpeTokenizerState SaveState()
        {
            return new SentencePieceBpeTokenizerState
            {
                Vocabulary = _vocabulary.SaveState(),
                Scores = new Dictionary<int, float>(_scores),
                TokenTypes = new Dictionary<int, byte>(_tokenTypes),
                AddDummyPrefix = _addDummyPrefix,
                PadToken = PadToken,
                UnknownToken = UnknownToken,
                StartToken = StartToken,
                EndToken = EndToken,
                SepToken = SepToken,
                ImStartToken = ImStartToken,
                ImEndToken = ImEndToken
            };
        }

        public static SentencePieceBpeTokenizer LoadState(SentencePieceBpeTokenizerState state)
        {
            var tokenizer = new SentencePieceBpeTokenizer
            {
                _vocabulary = Vocabulary.LoadState(state.Vocabulary),
                _scores = new Dictionary<int, float>(state.Scores),
                _tokenTypes = new Dictionary<int, byte>(state.TokenTypes),
                _addDummyPrefix = state.AddDummyPrefix,
                PadToken = state.PadToken,
                UnknownToken = state.UnknownToken,
                StartToken = state.StartToken,
                EndToken = state.EndToken,
                SepToken = state.SepToken,
                ImStartToken = state.ImStartToken,
                ImEndToken = state.ImEndToken
            };

            tokenizer._byteTokenId = BuildByteTokenLookup(tokenizer._vocabulary);

            return tokenizer;
        }

        private static Dictionary<byte, int> BuildByteTokenLookup(Vocabulary vocabulary)
        {
            var lookup = new Dictionary<byte, int>();

            for (int b = 0; b <= 0xFF; b++)
            {
                var token = $"<0x{b:X2}>";
                if (vocabulary.TryGetId(token, out var id))
                    lookup[(byte)b] = id;
            }

            return lookup;
        }
    }

    public sealed class SentencePieceBpeTokenizerState
    {
        public VocabularyState Vocabulary { get; set; } = new();
        public Dictionary<int, float> Scores { get; set; } = new();
        public Dictionary<int, byte> TokenTypes { get; set; } = new();
        public bool AddDummyPrefix { get; set; } = true;
        public int PadToken { get; set; }
        public int UnknownToken { get; set; }
        public int StartToken { get; set; }
        public int EndToken { get; set; }
        public int SepToken { get; set; }
        public int ImStartToken { get; set; }
        public int ImEndToken { get; set; }
    }
}
