namespace Neuraval.Core.Tokenizers
{
    public sealed class Vocabulary
    {
        private Dictionary<string, int> _tokenToId = new();
        private Dictionary<int, string> _idToToken = new();

        public int Count => _tokenToId.Count;

        public IReadOnlyDictionary<string, int> TokenToId => _tokenToId;

        public IReadOnlyDictionary<int, string> IdToToken => _idToToken;

        public int AddToken(string token)
        {
            if (_tokenToId.TryGetValue(token, out var existingId))
                return existingId;

            int id = _tokenToId.Count;
            _tokenToId[token] = id;
            _idToToken[id] = token;
            return id;
        }

        public bool Contains(string token)
        {
            return _tokenToId.ContainsKey(token);
        }

        public bool TryGetId(string token, out int id)
        {
            return _tokenToId.TryGetValue(token, out id);
        }

        public int GetId(string token, int unknownId)
        {
            return _tokenToId.TryGetValue(token, out var id) ? id : unknownId;
        }

        public string GetToken(int id, string unknownToken)
        {
            return _idToToken.TryGetValue(id, out var token) ? token : unknownToken;
        }

        public VocabularyState SaveState()
        {
            return new VocabularyState
            {
                TokenToId = new Dictionary<string, int>(_tokenToId)
            };
        }

        public static Vocabulary LoadState(VocabularyState state)
        {
            var vocabulary = new Vocabulary();

            foreach (var pair in state.TokenToId.OrderBy(p => p.Value))
            {
                vocabulary._tokenToId[pair.Key] = pair.Value;
                vocabulary._idToToken[pair.Value] = pair.Key;
            }

            return vocabulary;
        }
    }

    public sealed class VocabularyState
    {
        public Dictionary<string, int> TokenToId { get; set; } = new();
    }
}
