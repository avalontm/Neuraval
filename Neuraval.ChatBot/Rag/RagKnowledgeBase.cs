using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace Neuraval.ChatBot.Rag
{
    public sealed record RagChunk(string Source, int Part, string Text);

    public sealed record RagSearchResult(RagChunk Chunk, double Score);

    /// <summary>Local BM25 index for plain-text and Markdown files.</summary>
    public sealed class RagKnowledgeBase
    {
        private const int DefaultChunkCharacters = 1200;
        private const double K1 = 1.5;
        private const double B = 0.75;
        private static readonly Regex TokenPattern = new(@"[\p{L}\p{N}]{2,}", RegexOptions.Compiled);

        private readonly List<RagChunk> _chunks;
        private readonly List<string[]> _tokens;
        private readonly List<Dictionary<string, int>> _termFrequencies;
        private readonly Dictionary<string, int> _documentFrequencies;
        private readonly double _averageLength;

        public int DocumentCount { get; }
        public int ChunkCount => _chunks.Count;

        private RagKnowledgeBase(List<RagChunk> chunks, int documentCount)
        {
            _chunks = chunks;
            DocumentCount = documentCount;
            _tokens = chunks.Select(chunk => Tokenize(chunk.Text)).ToList();
            _termFrequencies = new List<Dictionary<string, int>>(_tokens.Count);
            _documentFrequencies = new Dictionary<string, int>(StringComparer.Ordinal);

            foreach (var terms in _tokens)
            {
                var frequencies = new Dictionary<string, int>(StringComparer.Ordinal);
                foreach (var term in terms)
                {
                    frequencies[term] = frequencies.GetValueOrDefault(term) + 1;
                }

                _termFrequencies.Add(frequencies);
                foreach (var term in frequencies.Keys)
                {
                    _documentFrequencies[term] = _documentFrequencies.GetValueOrDefault(term) + 1;
                }
            }

            _averageLength = _tokens.Count == 0 ? 0 : _tokens.Average(terms => terms.Length);
        }

        public static RagKnowledgeBase LoadDirectory(string directory, int chunkCharacters = DefaultChunkCharacters)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(directory);
            if (chunkCharacters < 100)
            {
                throw new ArgumentOutOfRangeException(nameof(chunkCharacters), "El tamaño mínimo de fragmento es 100 caracteres.");
            }

            var root = Path.GetFullPath(directory);
            if (!Directory.Exists(root))
            {
                throw new DirectoryNotFoundException($"No existe la carpeta de documentos RAG: {root}");
            }

            var files = Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories)
                .Where(path => Path.GetExtension(path).Equals(".txt", StringComparison.OrdinalIgnoreCase)
                    || Path.GetExtension(path).Equals(".md", StringComparison.OrdinalIgnoreCase))
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            var chunks = new List<RagChunk>();
            foreach (var file in files)
            {
                var text = File.ReadAllText(file);
                var source = Path.GetRelativePath(root, file).Replace(Path.DirectorySeparatorChar, '/');
                AddChunks(chunks, source, text, chunkCharacters);
            }

            if (chunks.Count == 0)
            {
                throw new InvalidOperationException($"No encontré archivos .txt o .md con contenido en '{root}'.");
            }

            return new RagKnowledgeBase(chunks, files.Length);
        }

        public IReadOnlyList<RagSearchResult> Search(string query, int topK = 3)
        {
            if (topK <= 0 || _chunks.Count == 0)
            {
                return Array.Empty<RagSearchResult>();
            }

            var queryTerms = Tokenize(query).Distinct(StringComparer.Ordinal).ToArray();
            if (queryTerms.Length == 0 || _averageLength <= 0)
            {
                return Array.Empty<RagSearchResult>();
            }

            var results = new List<RagSearchResult>();
            for (var index = 0; index < _chunks.Count; index++)
            {
                var frequencies = _termFrequencies[index];
                var length = _tokens[index].Length;
                double score = 0;

                foreach (var term in queryTerms)
                {
                    if (!frequencies.TryGetValue(term, out var frequency)
                        || !_documentFrequencies.TryGetValue(term, out var documentFrequency))
                    {
                        continue;
                    }

                    var idf = Math.Log(1 + (_chunks.Count - documentFrequency + 0.5) / (documentFrequency + 0.5));
                    var denominator = frequency + K1 * (1 - B + B * length / _averageLength);
                    score += idf * frequency * (K1 + 1) / denominator;
                }

                if (score > 0)
                {
                    results.Add(new RagSearchResult(_chunks[index], score));
                }
            }

            return results.OrderByDescending(result => result.Score).Take(topK).ToArray();
        }

        private static void AddChunks(List<RagChunk> chunks, string source, string text, int maxCharacters)
        {
            var normalized = text.Replace("\r\n", "\n", StringComparison.Ordinal).Trim();
            if (normalized.Length == 0)
            {
                return;
            }

            var part = 0;
            var pending = new StringBuilder();
            foreach (var paragraph in Regex.Split(normalized, @"\n\s*\n"))
            {
                var value = paragraph.Trim();
                if (value.Length == 0)
                {
                    continue;
                }

                if (value.Length > maxCharacters)
                {
                    FlushPending();
                    var start = 0;
                    while (start < value.Length)
                    {
                        var end = Math.Min(start + maxCharacters, value.Length);
                        if (end < value.Length)
                        {
                            var boundary = value.LastIndexOfAny(new[] { ' ', '\n', '\t' }, end - 1, end - start);
                            if (boundary > start + maxCharacters / 2)
                            {
                                end = boundary;
                            }
                        }

                        var slice = value[start..end].Trim();
                        if (slice.Length > 0)
                        {
                            chunks.Add(new RagChunk(source, part++, slice));
                        }

                        start = end;
                        while (start < value.Length && char.IsWhiteSpace(value[start])) start++;
                    }

                    continue;
                }

                if (pending.Length > 0 && pending.Length + value.Length + 2 > maxCharacters)
                {
                    FlushPending();
                }

                if (pending.Length > 0)
                {
                    pending.AppendLine().AppendLine();
                }

                pending.Append(value);
            }

            FlushPending();

            void FlushPending()
            {
                if (pending.Length == 0)
                {
                    return;
                }

                chunks.Add(new RagChunk(source, part++, pending.ToString()));
                pending.Clear();
            }
        }

        private static string[] Tokenize(string text)
        {
            var decomposed = text.Normalize(NormalizationForm.FormD);
            var normalized = new StringBuilder(decomposed.Length);
            foreach (var character in decomposed)
            {
                if (CharUnicodeInfo.GetUnicodeCategory(character) != UnicodeCategory.NonSpacingMark)
                {
                    normalized.Append(char.ToLowerInvariant(character));
                }
            }

            return TokenPattern.Matches(normalized.ToString())
                .Select(match => match.Value)
                .ToArray();
        }
    }
}
