namespace Neuraval.Core.Services
{
    public class Tokenizer : ITokenizer
    {
        private Dictionary<string, int> _tokenToId;
        private Dictionary<int, string> _idToToken;
        private int _nextId;

        public int VocabSize => _tokenToId.Count;
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

        public Tokenizer()
        {
            _tokenToId = new Dictionary<string, int>();
            _idToToken = new Dictionary<int, string>();
            _nextId = 0;

            PadToken = AddSpecialToken(PAD);
            UnknownToken = AddSpecialToken(UNK);
            StartToken = AddSpecialToken(START);
            EndToken = AddSpecialToken(END);
            SepToken = AddSpecialToken(SEP);
        }

        private int AddSpecialToken(string token)
        {
            if (!_tokenToId.ContainsKey(token))
            {
                _tokenToId[token] = _nextId;
                _idToToken[_nextId] = token;
                _nextId++;
            }
            return _tokenToId[token];
        }

        public void BuildVocabulary(List<string> texts)
        {
            BuildVocabulary(texts, minFrequency: 1, maxVocabSize: 50000);
        }

        public void BuildVocabulary(List<string> texts, int vocabSize)
        {
            BuildVocabulary(texts, minFrequency: 1, maxVocabSize: vocabSize);
        }

        public void BuildVocabulary(List<string> texts, int minFrequency = 1, int maxVocabSize = 50000)
        {
            var tokenFrequency = new Dictionary<string, int>();

            foreach (var text in texts)
            {
                var tokens = TokenizeText(text);
                foreach (var token in tokens)
                {
                    if (!tokenFrequency.ContainsKey(token))
                    {
                        tokenFrequency[token] = 0;
                    }
                    tokenFrequency[token]++;
                }
            }

            var sortedTokens = tokenFrequency
                .Where(kvp => kvp.Value >= minFrequency)
                .OrderByDescending(kvp => kvp.Value)
                .Take(maxVocabSize - _nextId)
                .Select(kvp => kvp.Key);

            foreach (var token in sortedTokens)
            {
                AddToken(token);
            }

            System.Console.WriteLine($"Vocabulary built: {VocabSize} tokens");
        }

        public int AddToken(string token)
        {
            if (_tokenToId.ContainsKey(token))
            {
                return _tokenToId[token];
            }

            _tokenToId[token] = _nextId;
            _idToToken[_nextId] = token;
            _nextId++;

            return _nextId - 1;
        }

        public int[] Encode(string text, bool addSpecialTokens = true)
        {
            var tokens = TokenizeText(text);
            var ids = new List<int>();

            if (addSpecialTokens)
            {
                ids.Add(StartToken);
            }

            foreach (var token in tokens)
            {
                if (_tokenToId.ContainsKey(token))
                {
                    ids.Add(_tokenToId[token]);
                }
                else
                {
                    ids.Add(UnknownToken);
                }
            }

            if (addSpecialTokens)
            {
                ids.Add(EndToken);
            }

            return ids.ToArray();
        }

        public string Decode(int[] tokenIds, bool skipSpecialTokens = true)
        {
            var tokens = new List<string>();

            foreach (var id in tokenIds)
            {
                if (_idToToken.ContainsKey(id))
                {
                    var token = _idToToken[id];

                    if (skipSpecialTokens && IsSpecialToken(token))
                    {
                        continue;
                    }

                    tokens.Add(token);
                }
            }

            return string.Join(" ", tokens);
        }

        private bool IsSpecialToken(string token)
        {
            return token == PAD || token == UNK || token == START || token == END || token == SEP;
        }

        public int[] EncodeCausalSequence(string prompt, string response)
        {
            var promptTokens = TokenizeText(prompt);
            var responseTokens = TokenizeText(response);

            var ids = new List<int> { StartToken };

            foreach (var token in promptTokens)
            {
                ids.Add(_tokenToId.TryGetValue(token, out var promptId) ? promptId : UnknownToken);
            }

            ids.Add(SepToken);

            foreach (var token in responseTokens)
            {
                ids.Add(_tokenToId.TryGetValue(token, out var responseId) ? responseId : UnknownToken);
            }

            ids.Add(EndToken);

            return ids.ToArray();
        }

        public int[] EncodePrompt(string prompt)
        {
            var promptTokens = TokenizeText(prompt);
            var ids = new List<int> { StartToken };

            foreach (var token in promptTokens)
            {
                ids.Add(_tokenToId.TryGetValue(token, out var promptId) ? promptId : UnknownToken);
            }

            ids.Add(SepToken);

            return ids.ToArray();
        }

        private List<string> TokenizeText(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
            {
                return new List<string>();
            }

            text = text.ToLower();
            text = System.Text.RegularExpressions.Regex.Replace(text, @"[^\w\s]", " $0 ");
            text = System.Text.RegularExpressions.Regex.Replace(text, @"\s+", " ");

            var tokens = text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(t => t.Trim())
                .Where(t => !string.IsNullOrEmpty(t))
                .ToList();

            return tokens;
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
            return _tokenToId.ContainsKey(token) ? _tokenToId[token] : UnknownToken;
        }

        public string GetToken(int id)
        {
            return _idToToken.ContainsKey(id) ? _idToToken[id] : UNK;
        }

        public TokenizerState SaveState()
        {
            return new TokenizerState
            {
                TokenToId = new Dictionary<string, int>(_tokenToId),
                IdToToken = new Dictionary<int, string>(_idToToken),
                NextId = _nextId,
                PadToken = PadToken,
                UnknownToken = UnknownToken,
                StartToken = StartToken,
                EndToken = EndToken,
                SepToken = SepToken
            };
        }

        public static Tokenizer LoadState(TokenizerState state)
        {
            var tokenizer = new Tokenizer();
            tokenizer._tokenToId = new Dictionary<string, int>(state.TokenToId);
            tokenizer._idToToken = new Dictionary<int, string>(state.IdToToken);
            tokenizer._nextId = state.NextId;
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

        public static Tokenizer LoadFromFile(string filepath)
        {
            if (!File.Exists(filepath))
            {
                throw new FileNotFoundException($"Tokenizer file not found: {filepath}");
            }

            var json = File.ReadAllText(filepath);
            var state = System.Text.Json.JsonSerializer.Deserialize<TokenizerState>(json);

            if (state == null)
            {
                throw new InvalidOperationException("Failed to deserialize tokenizer state");
            }

            return LoadState(state);
        }

        public Dictionary<string, int> GetVocabulary()
        {
            return new Dictionary<string, int>(_tokenToId);
        }

        public List<string> GetAllTokens()
        {
            return _idToToken.OrderBy(kvp => kvp.Key).Select(kvp => kvp.Value).ToList();
        }

        public int GetFrequency(string token, List<string> texts)
        {
            int count = 0;
            foreach (var text in texts)
            {
                var tokens = TokenizeText(text);
                count += tokens.Count(t => t == token);
            }
            return count;
        }
    }

    public class TokenizerState
    {
        public Dictionary<string, int> TokenToId { get; set; }
        public Dictionary<int, string> IdToToken { get; set; }
        public int NextId { get; set; }
        public int PadToken { get; set; }
        public int UnknownToken { get; set; }
        public int StartToken { get; set; }
        public int EndToken { get; set; }
        public int SepToken { get; set; }

        public TokenizerState()
        {
            TokenToId = new Dictionary<string, int>();
            IdToToken = new Dictionary<int, string>();
        }
    }
}