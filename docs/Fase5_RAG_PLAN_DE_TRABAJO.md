# Fase 5 — RAG (consulta de documentos tipo agente)

> Se agrega como Fase 5 de `PLAN_DE_TRABAJO.md`, siguiendo la misma regla del resto del plan: no se avanza de sub-fase sin `dotnet test` en verde confirmado por vos en tu equipo.

## Objetivo de la fase

Que NeuralChatBot pueda responder preguntas usando documentos externos como contexto, en vez de depender solo de lo memorizado durante el entrenamiento. El modelo no cambia de arquitectura; se le agrega una capa de **retrieval** (búsqueda) antes de generar la respuesta.

```
Pregunta
   │
   ▼
Índice de documentos (chunks)
   │
   ▼
Top-K fragmentos relevantes
   │
   ▼
Prompt = [fragmentos] + [pregunta]
   │
   ▼
NeuralChatBot genera la respuesta
```

## Supuestos sobre tu código actual

Los snippets abajo asumen (ajustá nombres si difieren):
- Ya tenés `ITokenizer` con `Encode(string) -> int[]` / `Decode(int[]) -> string` (BPE, de la Fase 3)
- Ya tenés un modelo transformer con un método tipo `Generate(int[] promptTokens, int maxNewTokens) -> int[]`
- El proyecto principal es `NeuralChatBot` y los tests en `NeuralChatBot.Tests`

Si algún nombre no coincide, son cambios triviales de renombrado — la lógica no depende de los nombres exactos.

---

## Sub-fase 5.1 — Chunking de documentos

**Objetivo:** partir documentos largos en fragmentos manejables antes de indexarlos.

```csharp
namespace NeuralChatBot.Rag
{
    public sealed class DocumentChunk
    {
        public required string Id { get; init; }
        public required string SourceDocument { get; init; }
        public required string Text { get; init; }
    }

    public sealed class ChunkingOptions
    {
        // Tamaño objetivo del chunk, en tokens (no caracteres)
        public int ChunkSizeTokens { get; init; } = 200;
        // Solapamiento entre chunks consecutivos, para no cortar ideas a la mitad
        public int OverlapTokens { get; init; } = 40;
    }

    public static class DocumentChunker
    {
        public static List<DocumentChunk> ChunkDocument(
            string documentId,
            string fullText,
            ITokenizer tokenizer,
            ChunkingOptions options)
        {
            var tokens = tokenizer.Encode(fullText);
            var chunks = new List<DocumentChunk>();

            var start = 0;
            var chunkIndex = 0;

            while (start < tokens.Length)
            {
                var end = Math.Min(start + options.ChunkSizeTokens, tokens.Length);
                var slice = tokens[start..end];
                var text = tokenizer.Decode(slice);

                chunks.Add(new DocumentChunk
                {
                    Id = $"{documentId}#{chunkIndex}",
                    SourceDocument = documentId,
                    Text = text
                });

                if (end == tokens.Length) break;

                start = end - options.OverlapTokens; // retrocede para el solapamiento
                chunkIndex++;
            }

            return chunks;
        }
    }
}
```

**Test a agregar (`ChunkingTests.cs`):**
- Un documento corto (< ChunkSizeTokens) produce exactamente 1 chunk
- Un documento largo produce N chunks con solapamiento correcto
- No se pierden tokens al final del documento (el último chunk llega hasta el final)

**Criterio de salida 5.1:** chunking determinista y con tests en verde, sin retrieval todavía.

---

## Sub-fase 5.2 — Índice de retrieval léxico (BM25, sin red neuronal)

Empezar por acá y no por embeddings: es más simple, no depende de tu modelo entrenado, y ya da resultados útiles.

```csharp
namespace NeuralChatBot.Rag
{
    public sealed class Bm25Index
    {
        private readonly List<DocumentChunk> _chunks = new();
        private readonly List<string[]> _tokenizedChunks = new();
        private readonly Dictionary<string, int> _documentFrequency = new();
        private double _averageDocLength;

        private const double K1 = 1.5;
        private const double B = 0.75;

        public void AddChunks(IEnumerable<DocumentChunk> chunks)
        {
            foreach (var chunk in chunks)
            {
                var terms = Tokenize(chunk.Text);
                _chunks.Add(chunk);
                _tokenizedChunks.Add(terms);

                foreach (var term in terms.Distinct())
                {
                    _documentFrequency[term] = _documentFrequency.GetValueOrDefault(term) + 1;
                }
            }

            _averageDocLength = _tokenizedChunks.Average(t => t.Length);
        }

        public List<(DocumentChunk Chunk, double Score)> Search(string query, int topK)
        {
            var queryTerms = Tokenize(query);
            var scores = new double[_chunks.Count];
            var n = _chunks.Count;

            for (var i = 0; i < n; i++)
            {
                var docTerms = _tokenizedChunks[i];
                var docLength = docTerms.Length;
                var termCounts = docTerms.GroupBy(t => t).ToDictionary(g => g.Key, g => g.Count());

                foreach (var qt in queryTerms)
                {
                    if (!termCounts.TryGetValue(qt, out var tf)) continue;

                    var df = _documentFrequency.GetValueOrDefault(qt, 0);
                    if (df == 0) continue;

                    var idf = Math.Log(1 + (n - df + 0.5) / (df + 0.5));
                    var numerator = tf * (K1 + 1);
                    var denominator = tf + K1 * (1 - B + B * docLength / _averageDocLength);

                    scores[i] += idf * (numerator / denominator);
                }
            }

            return Enumerable.Range(0, n)
                .Select(i => (_chunks[i], scores[i]))
                .OrderByDescending(x => x.Item2)
                .Take(topK)
                .ToList();
        }

        private static string[] Tokenize(string text) =>
            text.ToLowerInvariant()
                .Split(new[] { ' ', '\n', '\r', '\t', '.', ',', ';', ':', '!', '?', '"', '(', ')' },
                       StringSplitOptions.RemoveEmptyEntries);
    }
}
```

**Test a agregar (`Bm25IndexTests.cs`):**
- Un query con términos exclusivos de un chunk devuelve ese chunk primero
- `topK` respeta el límite pedido
- Query sin coincidencias devuelve lista vacía o scores en 0, sin excepción

**Criterio de salida 5.2:** dado un conjunto de documentos de prueba, el índice devuelve los chunks correctos para preguntas simples, verificado por tests.

---

## Sub-fase 5.3 — (Opcional) Embeddings propios usando NeuralChatBot

Mejora semántica sobre BM25 (encuentra sinónimos/paráfrasis que BM25 no detecta por coincidencia literal de palabras). Se puede dejar para después de tener 5.2 funcionando y evaluar si hace falta.

```csharp
namespace NeuralChatBot.Rag
{
    public sealed class EmbeddingIndex
    {
        private readonly List<(DocumentChunk Chunk, float[] Vector)> _entries = new();
        private readonly ITransformerEncoder _model; // tu modelo, expuesto para sacar hidden states

        public EmbeddingIndex(ITransformerEncoder model) => _model = model;

        public void AddChunk(DocumentChunk chunk, ITokenizer tokenizer)
        {
            var tokens = tokenizer.Encode(chunk.Text);
            var vector = _model.GetMeanPooledHiddenState(tokens); // promedio de la última capa oculta
            _entries.Add((chunk, Normalize(vector)));
        }

        public List<(DocumentChunk Chunk, double Score)> Search(string query, ITokenizer tokenizer, int topK)
        {
            var queryTokens = tokenizer.Encode(query);
            var queryVector = Normalize(_model.GetMeanPooledHiddenState(queryTokens));

            return _entries
                .Select(e => (e.Chunk, Score: (double)CosineSimilarity(queryVector, e.Vector)))
                .OrderByDescending(x => x.Score)
                .Take(topK)
                .ToList();
        }

        private static float[] Normalize(float[] v)
        {
            var norm = MathF.Sqrt(v.Sum(x => x * x));
            return norm < 1e-8f ? v : v.Select(x => x / norm).ToArray();
        }

        private static float CosineSimilarity(float[] a, float[] b) =>
            a.Zip(b, (x, y) => x * y).Sum(); // ya normalizados => producto punto = coseno
    }
}
```

> **Nota:** esto requiere exponer un método nuevo en tu motor (`GetMeanPooledHiddenState`) que corra el forward pass y devuelva el promedio de la última capa oculta en vez de generar tokens. Es un cambio quirúrgico, no toca el resto del modelo.

**Criterio de salida 5.3 (si se hace):** comparar recall de BM25 vs. embeddings en un set de preguntas de prueba con paráfrasis; solo vale la pena mantenerlo si mejora resultados medibles.

---

## Sub-fase 5.4 — Construcción del prompt con contexto inyectado

```csharp
namespace NeuralChatBot.Rag
{
    public static class RagPromptBuilder
    {
        public static string BuildPrompt(string userQuestion, IReadOnlyList<DocumentChunk> retrievedChunks)
        {
            var sb = new StringBuilder();
            sb.AppendLine("Contexto:");

            foreach (var chunk in retrievedChunks)
            {
                sb.AppendLine($"[{chunk.SourceDocument}] {chunk.Text}");
            }

            sb.AppendLine();
            sb.AppendLine($"Pregunta: {userQuestion}");
            sb.AppendLine("Respuesta:");

            return sb.ToString();
        }
    }
}
```

**Punto importante:** este formato exacto (`Contexto: ... Pregunta: ... Respuesta:`) tiene que coincidir con el formato usado en el fine-tuning de la sub-fase 5.5. Si el formato de entrenamiento y el de inferencia no coinciden, el modelo no va a "aprender" a usar el contexto de forma consistente.

**Test a agregar:** el prompt generado contiene todos los chunks pasados y termina siempre en `"Respuesta:"`.

**Criterio de salida 5.4:** prompt determinista y testeado; todavía sin fine-tuning, solo construcción de texto.

---

## Sub-fase 5.5 — Fine-tuning para uso de contexto (grounded QA)

Esta es la sub-fase que realmente le enseña al modelo a **priorizar el contexto** por sobre lo que "cree recordar" del pre-entrenamiento general.

**Formato del dataset de fine-tuning** (mismo formato que `RagPromptBuilder`):

```
Contexto:
[doc1] <fragmento real>
[doc2] <fragmento real>

Pregunta: <pregunta cuya respuesta está en los fragmentos de arriba>
Respuesta: <respuesta extraída/derivada del contexto>
```

**Cómo generar los pares de entrenamiento sin trabajo manual enorme:**
1. Tomar documentos que ya tengas
2. Generar preguntas automáticas sobre cada chunk (podés generarlas vos mismo a mano para un set inicial chico, o con reglas simples tipo "¿Qué es X?" a partir de definiciones detectadas)
3. La respuesta correcta es literalmente el fragmento (o una versión resumida) — no hace falta creatividad del modelo acá, solo que aprenda a **copiar/extraer** del contexto

**Importante — incluir también ejemplos negativos:** algunos pares donde la pregunta **no** se puede responder con el contexto dado, y la respuesta esperada sea algo como `"No tengo esa información en el contexto proporcionado."`. Sin esto, el modelo tiende a alucinar una respuesta igual aunque el contexto no la contenga.

**Criterio de salida 5.5:** `dotnet test` en verde para el pipeline de fine-tuning + evaluación manual sobre un set de preguntas conocidas (¿el modelo responde con lo que está en el contexto, o inventa?).

---

## Sub-fase 5.6 — Integración en modo interactivo ("agente")

```csharp
namespace NeuralChatBot.Rag
{
    public sealed class DocumentAgent
    {
        private readonly Bm25Index _index;
        private readonly ITokenizer _tokenizer;
        private readonly IChatModel _model; // tu modelo con Generate(...)

        public DocumentAgent(Bm25Index index, ITokenizer tokenizer, IChatModel model)
        {
            _index = index;
            _tokenizer = tokenizer;
            _model = model;
        }

        public string Ask(string question, int topK = 3, int maxNewTokens = 150)
        {
            var results = _index.Search(question, topK);
            var chunks = results.Select(r => r.Chunk).ToList();

            var prompt = RagPromptBuilder.BuildPrompt(question, chunks);
            var promptTokens = _tokenizer.Encode(prompt);
            var outputTokens = _model.Generate(promptTokens, maxNewTokens);

            return _tokenizer.Decode(outputTokens);
        }
    }
}
```

**Criterio de salida 5.6 (cierre de Fase 5):** desde tu CLI/loop de chat, poder cargar una carpeta de documentos, indexarlos, y hacer preguntas que se respondan citando/usando esos documentos — con tests de integración en verde.

---

## Checklist de la Fase 5

- [ ] 5.1 — Chunking de documentos (tests en verde)
- [ ] 5.2 — Índice BM25 léxico (tests en verde)
- [ ] 5.3 — Embeddings propios (opcional, solo si mejora medible sobre BM25)
- [ ] 5.4 — Construcción de prompt con contexto (tests en verde)
- [ ] 5.5 — Fine-tuning grounded QA + ejemplos negativos ("no sé")
- [ ] 5.6 — Integración en modo agente interactivo

Igual que en fases anteriores: no pasar a la siguiente sub-fase sin confirmar `dotnet test` en verde en tu equipo.
