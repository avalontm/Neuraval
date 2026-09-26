# Fase 22 — Compatibilidad GGUF

Namespace: `Neuraval.Core.Serialization.Gguf` (`Neuraval/Serialization/Gguf/`).

## Alcance

GGUF como via de interoperabilidad de entrada, tal como pide el roadmap: no se reescribe
llama.cpp ni se agrega un writer GGUF, solo el loader.

```
GGUF (.gguf)
   |
GgufReader            -> header, metadata tipada, tensor infos, alineacion, dequantizacion
   |
GgufFile              -> metadata + tensores en float32
   |
GgufConfigMapper      -> TransformerConfig (con fallback a shapes de tensores si falta metadata)
GgufTensorNameMapper  -> nombres GGUF (blk.N.*, token_embd, output_norm, output) a
                          TensorNameMapper.* (model.layers.N.*, model.embed_tokens, ...)
   |
GgufModelLoader        -> SafeTensorsEntry[] + WeightManifestReport (reutiliza Fase 17/18)
```

## Tipos GGML soportados

`F32`, `F16`, `Q8_0`, `Q4_0` (dequantizados a `float[]` con la formula de referencia ggml,
bloques de 32 elementos). Otros tipos (`Q4_K`, `Q5_K`, etc.) lanzan `NotSupportedException`
explicito en vez de producir datos incorrectos.

## Componentes nuevos

- `GgufValueType.cs`, `GgmlType.cs` — enums del formato.
- `GgufMetadataValue.cs`, `GgufMetadataExtensions.cs` — acceso tipado a metadata GGUF.
- `GgufTensorEntry.cs`, `GgufFile.cs` — modelo en memoria.
- `GgufReader.cs` — parser binario completo (header, metadata incl. arrays, tensor infos,
  alineacion configurable via `general.alignment`, dequantizacion).
- `GgufTensorNameMapper.cs` — mapeo de nombres GGUF <-> `TensorNameMapper` existente.
- `GgufConfigMapper.cs` — metadata GGUF -> `TransformerConfig`, con fallback a shapes de
  tensores cuando falta una clave (`block_count`, `vocab_size`, `embedding_length`,
  `feed_forward_length`).
- `GgufModelLoader.cs` — orquesta lectura + mapeo + `TensorNameMapperValidator`.

## Tests

`Neuraval.Tests`: `GgufMetadataValueTests`, `GgufTensorNameMapperTests`,
`GgufConfigMapperTests`, `GgufModelLoaderTests`, `GgufReaderTests` (43 casos), cubriendo:
firma/version invalidas, cada tipo escalar de metadata, arrays de strings, decode exacto de
F32, tolerancia de F16, decode por bloque de Q8_0 y Q4_0, tipo no soportado, alineacion
custom con multiples tensores, resolucion de config con y sin fallback, integracion completa
del loader con embeddings atados y no atados.

No hubo acceso a `api.nuget.org` para restaurar `Neuraval.Tests` (mismo bloqueo de la sesion
anterior), asi que se validaron los 43 casos con un arnes standalone contra el `.csproj` real
de `Neuraval` — resultado: `ALL PASSED (53 checks)` (incluye variantes adicionales cubiertas
en el arnes).

## Siguiente

Fase 23 — Backend llama.cpp (`IChatModel` -> `LlamaCppChatModel` contra el servidor
OpenAI-compatible), o Fase 24 — backend OpenAI-compatible generico, segun lo que se priorice.
