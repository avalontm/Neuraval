# RAG local para el chat

El ejemplo agrega búsqueda BM25 local al chat existente. No usa servicios externos ni requiere entrenar embeddings. Se indexan archivos `.md` y `.txt` de forma recursiva y se recuperan hasta tres fragmentos por pregunta.

Desde la raíz del repositorio, ejecuta:

```powershell
dotnet run --project Neuraval.CLI -- chat --model "D:\Modelos\qwen2.5-0.5b-instruct-q8_0.gguf" --rag examples/rag/knowledge --stream
```

Reemplaza la ruta del modelo por tu archivo GGUF local. Para usar el modelo Neuraval guardado en `training-settings.json`, omite `--model`:

```powershell
dotnet run --project Neuraval.CLI -- chat --rag examples/rag/knowledge
```

Agrega tus documentos `.md` o `.txt` a la carpeta indicada. El CLI informa cuántos documentos y fragmentos indexó, y muestra las fuentes recuperadas para cada consulta. El agente inyecta esos fragmentos como contexto, solicita que no invente respuestas ausentes y pide citas con el nombre del archivo. El contexto sirve como apoyo de recuperación; la calidad final también depende del modelo local elegido.
