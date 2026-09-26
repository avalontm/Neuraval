# Fase 25 — Benchmarking

Roadmap: `Neuraval_LLM_Modernization_Roadmap.md`, sección "29. FASE 25 — Benchmarking".

## Objetivo del roadmap

Crear un sistema reproducible que compare Neuraval Native contra Qwen, Gemma, Llama y GPT/API,
midiendo tokens/sec, latencia, time to first token, RAM, VRAM, parameter count, model size, loss,
perplexity y un benchmark score.

## Qué se implementó

Nuevo namespace `Neuraval.ChatBot.Benchmarking` en `Neuraval.ChatBot/Benchmarking/`:

- `BenchmarkSubject` — un backend a comparar: nombre + `IChatModel` (Neuraval Native vía
  `TransformerChatBotService`, o cualquier modelo Qwen/Gemma/Llama/GPT servido por
  `LlamaCppChatModel`/`OpenAiCompatibleChatModel` de las Fases 23/24) + metadata estática opcional
  (parameter count, tamaño en disco).
- `ITokenCounter` con dos implementaciones: `WhitespaceTokenCounter` (aproximado, fallback para
  backends remotos) y `TokenizerTokenCounter` (exacto, delega en el `ITokenizer` del modelo nativo).
- `BenchmarkPromptResult` / `BenchmarkReport` — resultado por prompt y agregado por backend:
  tokens/sec, latencia promedio, éxitos/fallas, RAM del proceso.
- `BenchmarkRunner` — corre el mismo set de prompts, en el mismo orden, contra uno o varios
  `BenchmarkSubject`; si un prompt falla en un backend, se registra como falla y se sigue con el
  resto (no aborta la corrida completa).
- `BenchmarkScorer` — "benchmark score" relativo y transparente: normaliza tokens/sec y latencia
  promedio contra el mejor valor del grupo comparado (no hay una fórmula universal de score para
  LLMs; esta es explícita y documentada, no una caja negra).
- `BenchmarkReportFormatter` — tabla de texto para consola.
- CLI: `dotnet run --project Neuraval.CLI -- --chat-benchmark <config.json>` (`BenchmarkCliConfig`
  en `Neuraval.CLI/BenchmarkCliConfig.cs`), config JSON con `prompts` + `subjects` (backend
  `"native"`, `"llama.cpp"` o `"openai-compatible"`).

## Métricas del roadmap que quedaron explícitamente sin medir (no simuladas)

Siguiendo el mismo criterio que ya se usó en Neuraval (preferir datos verificados antes que
mapeos/números inventados):

- **Time to first token**: requiere respuestas en streaming; `IChatModel.SendAsync` devuelve el
  mensaje completo de una sola vez. `BenchmarkReport.TimeToFirstTokenSeconds` es siempre `null`.
- **VRAM**: requiere telemetría de GPU (`nvidia-smi` / `cudaMemGetInfo`), no disponible desde este
  proceso. Queda como `BenchmarkReport.VramBytes`, seteable desde afuera con `WithVram(...)` si el
  caller la mide por su cuenta.
- **Loss / Perplexity**: requieren un dataset de evaluación etiquetado, que es exactamente el
  alcance de la Fase 26 — Evaluación reproducible (la siguiente del roadmap). Quedan como
  `BenchmarkReport.Loss` / `Perplexity`, seteables con `WithEvaluationMetrics(...)` una vez que
  exista ese dataset.

## Tests

20 tests xUnit nuevos en `Neuraval.Tests/`: `BenchmarkRunnerTests`, `BenchmarkReportTests`,
`BenchmarkScorerTests`, `TokenCounterTests`, `BenchmarkReportFormatterTests`, más los helpers de
prueba `FakeChatModel` y `FakeTokenizer`.

## Compilación

Árbol completo compilado en este sandbox (.NET SDK 10 recién instalado): 0 errores en Neuraval,
Neuraval.Abstractions, Neuraval.Tensor, Neuraval.Cuda, Neuraval.ChatBot (incluyendo el nuevo
`Benchmarking/`), Neuraval.CLI, Neuraval.Evolution, Neuraval.Evolution.MarioBridge. Único fallo:
`Neuraval.Samples.DinoGame`, preexistente (targetea Windows).

Los tests xUnit nuevos no se corrieron en esta entrega (se pidió entregar el código sin compilar
ni ejecutar nada más); revisarlos/correrlos en el entorno con acceso a NuGet antes de dar la fase
por cerrada.
