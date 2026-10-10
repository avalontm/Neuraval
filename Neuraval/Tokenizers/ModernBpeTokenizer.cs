using Neuraval.Abstractions;
using Neuraval.Core.Services;

namespace Neuraval.Core.Tokenizers
{
    public sealed class ModernBpeTokenizer : IChatTokenizer
    {
        private static readonly System.Text.RegularExpressions.Regex Gpt2PreTokenPattern = new System.Text.RegularExpressions.Regex(
            @"\s?[\p{L}]+|\s?[\p{N}]+|\s?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            System.Text.RegularExpressions.RegexOptions.Compiled);
        private static readonly System.Text.RegularExpressions.Regex SmolLmPreTokenPattern = new System.Text.RegularExpressions.Regex(
            @"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            System.Text.RegularExpressions.RegexOptions.Compiled);
        private static readonly System.Text.RegularExpressions.Regex Qwen2PreTokenPattern = new System.Text.RegularExpressions.Regex(
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            System.Text.RegularExpressions.RegexOptions.Compiled);

        private Vocabulary _vocabulary;
        private List<BpeMergeRule> _merges;
        private Dictionary<(string First, string Second), int> _mergeRank;
        private string _preTokenizer = "gpt2";

        private Dictionary<int, byte> _tokenTypes = new();

        public int VocabSize => _vocabulary.Count;
        public int PadToken { get; private set; }
        public int UnknownToken { get; private set; }
        public int StartToken { get; private set; }
        public int EndToken { get; private set; }
        public bool AddBosToken { get; private set; } = true;
        public int SepToken { get; private set; }
        public int ImStartToken { get; private set; }
        public int ImEndToken { get; private set; }

        public ModernBpeTokenizer()
        {
            _vocabulary = new Vocabulary();
            _merges = new List<BpeMergeRule>();
            _mergeRank = new Dictionary<(string, string), int>();

            PadToken = _vocabulary.AddToken(SpecialTokens.Pad);
            UnknownToken = _vocabulary.AddToken(SpecialTokens.Unk);
            StartToken = _vocabulary.AddToken(SpecialTokens.Bos);
            EndToken = _vocabulary.AddToken(SpecialTokens.Eos);
            SepToken = _vocabulary.AddToken(SpecialTokens.Sep);
            ImStartToken = _vocabulary.AddToken(SpecialTokens.ImStart);
            ImEndToken = _vocabulary.AddToken(SpecialTokens.ImEnd);
        }

        public void BuildVocabulary(List<string> texts)
        {
            Train(texts, numMerges: 300, minPairFrequency: 2);
        }

        public void BuildVocabulary(List<string> texts, int vocabSize)
        {
            Train(texts, minPairFrequency: 2, targetVocabSize: vocabSize);
        }

        public void Train(List<string> texts, int numMerges = 300, int minPairFrequency = 2, int? targetVocabSize = null)
        {
            foreach (var symbol in ByteLevelEncoder.Alphabet)
                _vocabulary.AddToken(symbol.ToString());

            var chunkFrequency = new Dictionary<string, int>();

            foreach (var text in texts)
            {
                foreach (var chunk in PreTokenize(text))
                {
                    var encoded = ByteLevelEncoder.Encode(chunk);
                    chunkFrequency[encoded] = chunkFrequency.GetValueOrDefault(encoded, 0) + 1;
                }
            }

            var chunkSymbols = new Dictionary<string, List<string>>();

            foreach (var chunk in chunkFrequency.Keys)
                chunkSymbols[chunk] = chunk.Select(c => c.ToString()).ToList();

            int effectiveNumMerges = numMerges;

            if (targetVocabSize.HasValue)
                effectiveNumMerges = Math.Max(0, targetVocabSize.Value - _vocabulary.Count);

            _merges = new List<BpeMergeRule>();

            for (int step = 0; step < effectiveNumMerges; step++)
            {
                var pairCounts = new Dictionary<(string, string), int>();

                foreach (var chunk in chunkSymbols.Keys)
                {
                    var symbols = chunkSymbols[chunk];
                    var frequency = chunkFrequency[chunk];

                    for (int i = 0; i < symbols.Count - 1; i++)
                    {
                        var pair = (symbols[i], symbols[i + 1]);
                        pairCounts[pair] = pairCounts.GetValueOrDefault(pair, 0) + frequency;
                    }
                }

                if (pairCounts.Count == 0)
                    break;

                var bestPair = pairCounts
                    .OrderByDescending(kvp => kvp.Value)
                    .ThenBy(kvp => kvp.Key.Item1, StringComparer.Ordinal)
                    .ThenBy(kvp => kvp.Key.Item2, StringComparer.Ordinal)
                    .First();

                if (bestPair.Value < minPairFrequency)
                    break;

                var pairToMerge = bestPair.Key;
                var mergedSymbol = pairToMerge.Item1 + pairToMerge.Item2;

                _merges.Add(new BpeMergeRule { First = pairToMerge.Item1, Second = pairToMerge.Item2 });
                _vocabulary.AddToken(mergedSymbol);

                foreach (var chunk in chunkSymbols.Keys.ToList())
                    chunkSymbols[chunk] = ApplyMerge(chunkSymbols[chunk], pairToMerge);
            }

            RebuildMergeRank();
        }

        private void RebuildMergeRank()
        {
            _mergeRank = new Dictionary<(string, string), int>();

            for (int i = 0; i < _merges.Count; i++)
                _mergeRank[(_merges[i].First, _merges[i].Second)] = i;
        }

        private static List<string> ApplyMerge(List<string> symbols, (string First, string Second) pair)
        {
            var merged = new List<string>();
            int i = 0;

            while (i < symbols.Count)
            {
                if (i < symbols.Count - 1 && symbols[i] == pair.First && symbols[i + 1] == pair.Second)
                {
                    merged.Add(symbols[i] + symbols[i + 1]);
                    i += 2;
                }
                else
                {
                    merged.Add(symbols[i]);
                    i += 1;
                }
            }

            return merged;
        }

        public List<string> EncodeChunk(string encodedChunk)
        {
            var symbols = encodedChunk.Select(c => c.ToString()).ToList();

            while (symbols.Count > 1)
            {
                int bestRank = int.MaxValue;
                int bestIndex = -1;

                for (int i = 0; i < symbols.Count - 1; i++)
                {
                    if (_mergeRank.TryGetValue((symbols[i], symbols[i + 1]), out var rank) && rank < bestRank)
                    {
                        bestRank = rank;
                        bestIndex = i;
                    }
                }

                if (bestIndex < 0)
                    break;

                var pairToApply = (symbols[bestIndex], symbols[bestIndex + 1]);
                symbols = ApplyMerge(symbols, pairToApply);
            }

            return symbols;
        }

        private List<string> PreTokenize(string text)
        {
            if (string.IsNullOrEmpty(text))
                return new List<string>();

            var chunks = new List<string>();

            var pattern = string.Equals(_preTokenizer, "qwen2", StringComparison.OrdinalIgnoreCase)
                ? Qwen2PreTokenPattern
                : string.Equals(_preTokenizer, "smollm", StringComparison.OrdinalIgnoreCase)
                    ? SmolLmPreTokenPattern
                    : Gpt2PreTokenPattern;

            foreach (System.Text.RegularExpressions.Match match in pattern.Matches(text))
                chunks.Add(match.Value);

            return chunks;
        }

        private List<int> EncodeToIds(string text)
        {
            var ids = new List<int>();

            foreach (var chunk in PreTokenize(text))
            {
                var encodedChunk = ByteLevelEncoder.Encode(chunk);

                foreach (var symbol in EncodeChunk(encodedChunk))
                    ids.Add(_vocabulary.GetId(symbol, UnknownToken));
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
            return EncodeRenderedChatText(rendered);
        }

        private int[] EncodeRenderedChatText(string renderedText)
        {
            var specialTokens = BuildSpecialTokenLiterals();
            var ids = new List<int>();

            foreach (var segment in SpecialTokenTextSplitter.Split(renderedText, specialTokens))
            {
                if (segment.IsSpecial)
                    ids.Add(segment.TokenId);
                else
                    ids.AddRange(EncodeToIds(segment.Text));
            }

            if (AddBosToken && (ids.Count == 0 || ids[0] != StartToken))
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

            foreach (var id in _tokenTypes.Keys)
            {
                if (_tokenTypes[id] == 3)
                    AddIfPresent(id);
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

        public string Decode(int[] tokenIds, bool skipSpecialTokens = true)
        {
            var output = new System.Text.StringBuilder();
            var byteRun = new System.Text.StringBuilder();

            foreach (var id in tokenIds)
            {
                var token = _vocabulary.GetToken(id, SpecialTokens.Unk);

                if (IsSpecialOrControlToken(id))
                {
                    FlushByteRun(output, byteRun);

                    if (!skipSpecialTokens)
                        output.Append(token);

                    continue;
                }

                byteRun.Append(token);
            }

            FlushByteRun(output, byteRun);

            return output.ToString();
        }

        private bool IsSpecialOrControlToken(int id)
        {
            return id == PadToken || id == UnknownToken || id == StartToken || id == EndToken
                || id == SepToken || id == ImStartToken || id == ImEndToken
                || (_tokenTypes.TryGetValue(id, out var type) && type == 3);
        }

        private static void FlushByteRun(System.Text.StringBuilder output, System.Text.StringBuilder byteRun)
        {
            if (byteRun.Length == 0)
                return;

            output.Append(ByteLevelEncoder.Decode(byteRun.ToString()));
            byteRun.Clear();
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

        public ModernBpeTokenizerState SaveState()
        {
            return new ModernBpeTokenizerState
            {
                Vocabulary = _vocabulary.SaveState(),
                Merges = _merges.Select(m => new BpeMergeRule { First = m.First, Second = m.Second }).ToList(),
                TokenTypes = new Dictionary<int, byte>(_tokenTypes),
                PreTokenizer = _preTokenizer,
                AddBosToken = AddBosToken,
                PadToken = PadToken,
                UnknownToken = UnknownToken,
                StartToken = StartToken,
                EndToken = EndToken,
                SepToken = SepToken,
                ImStartToken = ImStartToken,
                ImEndToken = ImEndToken
            };
        }

        public static ModernBpeTokenizer LoadState(ModernBpeTokenizerState state)
        {
            var tokenizer = new ModernBpeTokenizer
            {
                _vocabulary = Vocabulary.LoadState(state.Vocabulary),
                _merges = state.Merges.Select(m => new BpeMergeRule { First = m.First, Second = m.Second }).ToList(),
                _tokenTypes = new Dictionary<int, byte>(state.TokenTypes),
                _preTokenizer = string.IsNullOrWhiteSpace(state.PreTokenizer) ? "gpt2" : state.PreTokenizer,
                AddBosToken = state.AddBosToken,
                PadToken = state.PadToken,
                UnknownToken = state.UnknownToken,
                StartToken = state.StartToken,
                EndToken = state.EndToken,
                SepToken = state.SepToken,
                ImStartToken = state.ImStartToken,
                ImEndToken = state.ImEndToken
            };

            tokenizer.RebuildMergeRank();

            return tokenizer;
        }

        public void SaveToFile(string filepath)
        {
            var state = SaveState();
            var json = System.Text.Json.JsonSerializer.Serialize(state, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(filepath, json);
        }

        public static ModernBpeTokenizer LoadFromFile(string filepath)
        {
            if (!File.Exists(filepath))
                throw new FileNotFoundException($"Archivo de tokenizer no encontrado: {filepath}");

            var json = File.ReadAllText(filepath);
            var state = System.Text.Json.JsonSerializer.Deserialize<ModernBpeTokenizerState>(json);

            if (state == null)
                throw new InvalidOperationException("No se pudo deserializar el estado del tokenizer");

            return LoadState(state);
        }
    }

    public sealed class BpeMergeRule
    {
        public string First { get; set; } = string.Empty;
        public string Second { get; set; } = string.Empty;
    }

    public sealed class ModernBpeTokenizerState
    {
        public VocabularyState Vocabulary { get; set; } = new();
        public List<BpeMergeRule> Merges { get; set; } = new();
        public Dictionary<int, byte> TokenTypes { get; set; } = new();
        public string PreTokenizer { get; set; } = "gpt2";
        public bool AddBosToken { get; set; } = true;
        public int PadToken { get; set; }
        public int UnknownToken { get; set; }
        public int StartToken { get; set; }
        public int EndToken { get; set; }
        public int SepToken { get; set; }
        public int ImStartToken { get; set; }
        public int ImEndToken { get; set; }
    }
}
