namespace Neuraval.Core.Services
{
    public class BpeTokenizer : ITokenizer
    {
        private Dictionary<string, int> _tokenToId;
        private Dictionary<int, string> _idToToken;
        private List<MergeRule> _merges;
        private Dictionary<(string First, string Second), int> _mergeRank;
        private int _nextId;

        public int VocabSize => _tokenToId.Count;
        public int MergeCount => _merges.Count;
        public int PadToken { get; private set; }
        public int UnknownToken { get; private set; }
        public int StartToken { get; private set; }
        public int EndToken { get; private set; }
        public int SepToken { get; private set; }

        public const string PAD = "<PAD>";
        public const string UNK = "<UNK>";
        public const string START = "<START>";
        public const string END = "<END>";
        public const string SEP = "<SEP>";
        public const string EndOfWord = "</w>";

        public BpeTokenizer()
        {
            _tokenToId = new Dictionary<string, int>();
            _idToToken = new Dictionary<int, string>();
            _merges = new List<MergeRule>();
            _mergeRank = new Dictionary<(string, string), int>();
            _nextId = 0;

            PadToken = AddToken(PAD);
            UnknownToken = AddToken(UNK);
            StartToken = AddToken(START);
            EndToken = AddToken(END);
            SepToken = AddToken(SEP);
        }

        public int AddToken(string token)
        {
            if (_tokenToId.TryGetValue(token, out var existingId))
            {
                return existingId;
            }

            _tokenToId[token] = _nextId;
            _idToToken[_nextId] = token;
            _nextId++;

            return _nextId - 1;
        }

        public void BuildVocabulary(List<string> texts)
        {
            Train(texts, numMerges: 300, minPairFrequency: 2);
        }

        public void Train(List<string> texts, int numMerges = 300, int minPairFrequency = 2)
        {
            var wordFrequency = new Dictionary<string, int>();

            foreach (var text in texts)
            {
                foreach (var word in PreTokenize(text))
                {
                    wordFrequency[word] = wordFrequency.GetValueOrDefault(word, 0) + 1;
                }
            }

            var wordSymbols = new Dictionary<string, List<string>>();

            foreach (var word in wordFrequency.Keys)
            {
                wordSymbols[word] = DecomposeIntoBaseSymbols(word);
            }

            var vocabSymbols = new HashSet<string>();
            foreach (var symbols in wordSymbols.Values)
            {
                foreach (var symbol in symbols)
                {
                    vocabSymbols.Add(symbol);
                }
            }

            _merges = new List<MergeRule>();

            for (int step = 0; step < numMerges; step++)
            {
                var pairCounts = new Dictionary<(string, string), int>();

                foreach (var word in wordSymbols.Keys)
                {
                    var symbols = wordSymbols[word];
                    var frequency = wordFrequency[word];

                    for (int i = 0; i < symbols.Count - 1; i++)
                    {
                        var pair = (symbols[i], symbols[i + 1]);
                        pairCounts[pair] = pairCounts.GetValueOrDefault(pair, 0) + frequency;
                    }
                }

                if (pairCounts.Count == 0)
                {
                    break;
                }

                var bestPair = pairCounts
                    .OrderByDescending(kvp => kvp.Value)
                    .ThenBy(kvp => kvp.Key.Item1, StringComparer.Ordinal)
                    .ThenBy(kvp => kvp.Key.Item2, StringComparer.Ordinal)
                    .First();

                if (bestPair.Value < minPairFrequency)
                {
                    break;
                }

                var pairToMerge = bestPair.Key;
                var mergedSymbol = pairToMerge.Item1 + pairToMerge.Item2;

                _merges.Add(new MergeRule { First = pairToMerge.Item1, Second = pairToMerge.Item2 });
                vocabSymbols.Add(mergedSymbol);

                foreach (var word in wordSymbols.Keys.ToList())
                {
                    wordSymbols[word] = ApplyMerge(wordSymbols[word], pairToMerge);
                }
            }

            RebuildMergeRank();

            foreach (var symbol in vocabSymbols.OrderBy(s => s, StringComparer.Ordinal))
            {
                AddToken(symbol);
            }
        }

        private void RebuildMergeRank()
        {
            _mergeRank = new Dictionary<(string, string), int>();
            for (int i = 0; i < _merges.Count; i++)
            {
                _mergeRank[(_merges[i].First, _merges[i].Second)] = i;
            }
        }

        private static List<string> DecomposeIntoBaseSymbols(string word)
        {
            var symbols = word.Select(c => c.ToString()).ToList();
            symbols[symbols.Count - 1] = symbols[^1] + EndOfWord;
            return symbols;
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

        public List<string> EncodeWord(string word)
        {
            var symbols = DecomposeIntoBaseSymbols(word);

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
                {
                    break;
                }

                var pairToApply = (symbols[bestIndex], symbols[bestIndex + 1]);
                symbols = ApplyMerge(symbols, pairToApply);
            }

            return symbols;
        }

        private List<string> PreTokenize(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return new List<string>();
            }

            text = text.ToLower();
            text = System.Text.RegularExpressions.Regex.Replace(text, @"[^\w\s]", " $0 ");
            text = System.Text.RegularExpressions.Regex.Replace(text, @"\s+", " ");

            return text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(t => t.Trim())
                .Where(t => !string.IsNullOrEmpty(t))
                .ToList();
        }

        private List<int> EncodeToIds(string text)
        {
            var ids = new List<int>();

            foreach (var word in PreTokenize(text))
            {
                foreach (var symbol in EncodeWord(word))
                {
                    ids.Add(_tokenToId.TryGetValue(symbol, out var id) ? id : UnknownToken);
                }
            }

            return ids;
        }

        public int[] Encode(string text, bool addSpecialTokens = true)
        {
            var ids = new List<int>();

            if (addSpecialTokens)
            {
                ids.Add(StartToken);
            }

            ids.AddRange(EncodeToIds(text));

            if (addSpecialTokens)
            {
                ids.Add(EndToken);
            }

            return ids.ToArray();
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
            var builder = new System.Text.StringBuilder();

            foreach (var id in tokenIds)
            {
                if (!_idToToken.TryGetValue(id, out var token))
                {
                    continue;
                }

                if (IsSpecialToken(token))
                {
                    if (!skipSpecialTokens)
                    {
                        builder.Append(token).Append(' ');
                    }
                    continue;
                }

                if (token.EndsWith(EndOfWord, StringComparison.Ordinal))
                {
                    builder.Append(token, 0, token.Length - EndOfWord.Length);
                    builder.Append(' ');
                }
                else
                {
                    builder.Append(token);
                }
            }

            return builder.ToString().Trim();
        }

        private bool IsSpecialToken(string token)
        {
            return token == PAD || token == UNK || token == START || token == END || token == SEP;
        }

        public int[] PadSequence(int[] sequence, int maxLength, bool padLeft = false)
        {
            if (sequence.Length >= maxLength)
            {
                return sequence.Take(maxLength).ToArray();
            }

            var padded = new int[maxLength];

            for (int i = 0; i < maxLength; i++)
            {
                padded[i] = PadToken;
            }

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
            return _tokenToId.ContainsKey(token);
        }

        public int GetTokenId(string token)
        {
            return _tokenToId.TryGetValue(token, out var id) ? id : UnknownToken;
        }

        public string GetToken(int id)
        {
            return _idToToken.TryGetValue(id, out var token) ? token : UNK;
        }

        public Dictionary<string, int> GetVocabulary()
        {
            return new Dictionary<string, int>(_tokenToId);
        }

        public List<string> GetAllTokens()
        {
            return _idToToken.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
        }

        public BpeTokenizerState SaveState()
        {
            return new BpeTokenizerState
            {
                TokenToId = new Dictionary<string, int>(_tokenToId),
                IdToToken = new Dictionary<int, string>(_idToToken),
                NextId = _nextId,
                Merges = _merges.Select(m => new MergeRule { First = m.First, Second = m.Second }).ToList(),
                PadToken = PadToken,
                UnknownToken = UnknownToken,
                StartToken = StartToken,
                EndToken = EndToken,
                SepToken = SepToken
            };
        }

        public static BpeTokenizer LoadState(BpeTokenizerState state)
        {
            var tokenizer = new BpeTokenizer();
            tokenizer._tokenToId = new Dictionary<string, int>(state.TokenToId);
            tokenizer._idToToken = new Dictionary<int, string>(state.IdToToken);
            tokenizer._nextId = state.NextId;
            tokenizer._merges = state.Merges.Select(m => new MergeRule { First = m.First, Second = m.Second }).ToList();
            tokenizer.RebuildMergeRank();
            tokenizer.PadToken = state.PadToken;
            tokenizer.UnknownToken = state.UnknownToken;
            tokenizer.StartToken = state.StartToken;
            tokenizer.EndToken = state.EndToken;
            tokenizer.SepToken = state.SepToken;

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

        public static BpeTokenizer LoadFromFile(string filepath)
        {
            if (!File.Exists(filepath))
            {
                throw new FileNotFoundException($"BPE tokenizer file not found: {filepath}");
            }

            var json = File.ReadAllText(filepath);
            var state = System.Text.Json.JsonSerializer.Deserialize<BpeTokenizerState>(json);

            if (state == null)
            {
                throw new InvalidOperationException("Failed to deserialize BPE tokenizer state");
            }

            return LoadState(state);
        }
    }

    public class MergeRule
    {
        public string First { get; set; } = string.Empty;
        public string Second { get; set; } = string.Empty;
    }

    public class BpeTokenizerState
    {
        public Dictionary<string, int> TokenToId { get; set; }
        public Dictionary<int, string> IdToToken { get; set; }
        public int NextId { get; set; }
        public List<MergeRule> Merges { get; set; }
        public int PadToken { get; set; }
        public int UnknownToken { get; set; }
        public int StartToken { get; set; }
        public int EndToken { get; set; }
        public int SepToken { get; set; }

        public BpeTokenizerState()
        {
            TokenToId = new Dictionary<string, int>();
            IdToToken = new Dictionary<int, string>();
            Merges = new List<MergeRule>();
        }
    }
}
