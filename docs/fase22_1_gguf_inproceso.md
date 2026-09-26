# Fase 22.1 — GGUF in-process (carga directa a ModernDecoderModel)

Continuación de la Fase 22 (`docs/fase22_gguf.md`). Esa fase deja `GgufModelLoader` produciendo
`SafeTensorsEntry[]` + `WeightManifestReport`, pero sin nada que arme un `ModernDecoderModel`
real a partir de esos tensores ni un `IChatModel` que corra inferencia 100% en proceso (sin
depender de un servidor `llama.cpp` externo, que es el camino que tomaron las Fases 23/24). Esta
sub-fase cierra ese hueco.

## Namespace

`Neuraval.Core.Serialization.Gguf` (`Neuraval/Serialization/Gguf/`) para los loaders nuevos, y
`Neuraval.ChatBot.Services` (`Neuraval.ChatBot/Services/`) para el `IChatModel`.

## Componentes nuevos

### `GgufModelWeightLoader.cs`

`GgufLoadResult` / `GgufFile` / path → `ModernDecoderModelState` → `ModernDecoderModel.LoadState(...)`.

Detalle crítico de orientación: `GgufReader.ToRowMajorShape` deja los pesos lineales en
convención `[out, in]` (como PyTorch/HF: `token_embd.weight` es `[vocab, hidden]`,
`ffn_gate.weight` es `[intermediate, hidden]`), pero las capas de Neuraval (`GQAAttention`,
`SwiGLUFeedForward`) guardan sus matrices como `[in, out]`. El loader transpone explícitamente
Q/K/V/O y gate/up/down; embedding y lm_head se copian directo porque ahí el layout ya coincide
(no son pesos de matmul en el sentido `[in,out]`, son tablas `[vocab,hidden]` en ambos formatos).

Antes de construir el estado valida que el manifiesto esté completo
(`GgufLoadResult.Report.IsComplete`) y que cada tensor tenga el shape esperado según la
`TransformerConfig` resuelta; cualquier tensor faltante o con shape incorrecto tira
`InvalidOperationException` con el nombre del tensor en el mensaje, nunca rellena con datos
inventados.

### `GgufTokenizerLoader.cs`

Reconstruye un `ModernBpeTokenizer` real desde `tokenizer.ggml.tokens` + `tokenizer.ggml.merges`
+ los `*_token_id`, usando `ModernBpeTokenizer.LoadState(...)` (bypaseando el constructor
público, que fija los 7 tokens especiales de Neuraval con IDs propios y pisaría los IDs reales
del GGUF).

Solo soporta tokenizers byte-level BPE (`tokenizer.ggml.model == "gpt2"`, que es lo que usan
Qwen/GPT-2-style); para SentencePiece (`llama`, etc.) tira `NotSupportedException` explícito en
vez de fingir que funciona.

`bos_token_id` y `eos_token_id` son obligatorios (si faltan, `InvalidOperationException`: sin
ellos no hay forma de saber dónde arranca ni dónde para una secuencia). `unknown_token_id`,
`padding_token_id` y `separator_token_id` son opcionales: si el GGUF no los declara, el loader
cae al ID de EOS. Esto es seguro porque esos tres campos solo los usan los métodos de
*training* de `ModernBpeTokenizer` (`PadSequence`, `EncodeCausalSequence`, etc.), que
`GgufChatModel` no ejercita; el camino de chat (`EncodeChat` → `Decode`) no depende de ellos.
`im_start_token_id` / `im_end_token_id` no existen como claves estándar de metadata GGUF, así que
se resuelven buscando el string literal (`<|im_start|>`, `<|im_end|>`) en el vocabulario cargado,
con fallback a BOS/EOS si el modelo no los declaró como tokens propios.

### `GgufChatModel.cs` + `GgufChatModelOptions`

`IChatModel` real: `GgufChatModel.FromFile(path)` (o el constructor directo con un
`ModernDecoderModel` + `ModernBpeTokenizer` ya cargados) que usa `TextGenerator.GenerateWithCache`
+ `ModernBpeTokenizer.EncodeChat` (plantilla ChatML por defecto, configurable vía
`GgufChatModelOptions.ChatTemplate`) para generar. Los tokens de parada son el EOS del tokenizer
y su `<|im_end|>` si el vocabulario lo define; ambos se filtran de la respuesta decodificada antes
de devolverla. `GgufChatModelOptions` expone temperatura, top-p/top-k, repetition penalty, modo
greedy y seed, en el mismo estilo que `LlamaCppOptions`/`OpenAiCompatibleOptions` de las Fases 23/24.

No se integró a `ChatModelFactory.CreateFromConfig` en esta pasada porque ese factory arma
backends *remotos* (URL + modelo) vía JSON, mientras que un modelo GGUF in-process se identifica
por un path a un archivo local; agregarlo requeriría una rama de config distinta (`"backend":
"gguf"` + `"path"`) que no estaba pedida explícitamente. Queda anotado como posible Fase 22.2 si
se necesita.

## Tests

`Neuraval.Tests`:

- `GgufModelWeightLoaderTests` (6 casos): tied/untied embeddings end-to-end (`Forward` sin
  NaN/Infinity), tensor faltante vía la API de alto nivel (`Load`) y vía `BuildState` directo,
  shape incorrecta, y un caso numérico que verifica la transposición exacta `[out,in]→[in,out]`
  contra valores conocidos.
- `GgufTokenizerLoaderTests` (9 casos): reconstrucción de vocabulario y merges, resolución de
  `im_start`/`im_end` por string literal, fallback de unknown/padding a EOS, valor declarado
  explícito de unknown, modelo SentencePiece no soportado, metadata faltante (`model`, `tokens`,
  `bos_token_id`, `eos_token_id`), y entrada de merge inválida.
- `GgufChatModelTests` (7 casos): greedy y sampling end-to-end contra un `ModernDecoderModel`
  chico entrenado con un tokenizer real (sin depender de un archivo `.gguf` en disco, para no
  necesitar un binario de prueba), mensajes vacíos, cancelación ya solicitada, argumentos nulos
  en el constructor, y archivo GGUF inexistente en `FromFile`.

## Compilación

No hubo acceso a un SDK de .NET en este sandbox (se intentó instalar `dotnet-sdk-10.0` vía apt;
los paquetes de `security.ubuntu.com` devuelven 404, mismo bloqueo que las fases anteriores). El
código se revisó a mano contra los tipos reales del repo pero no se compiló ni se corrieron los
tests. Correr `dotnet test` antes de dar la sub-fase por cerrada.

## Siguiente

- Fase 22.2 (opcional): integrar `GgufChatModel` a `ChatModelFactory`/`ChatBackendConfig` con un
  backend `"gguf"` basado en path local.
- Fase 26 — Evaluación reproducible (siguiente ítem real del roadmap, ya anotado como pendiente
  en `docs/fase25_benchmarking.md`).
