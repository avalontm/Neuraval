# Neuraval: ejemplo de conocimiento local

Neuraval es un proyecto .NET para experimentar con modelos de lenguaje, tensores y entrenamiento local. El chat del CLI puede cargar modelos GGUF locales o modelos Neuraval guardados.

El comando `chat --model <archivo.gguf>` abre un modelo GGUF local. Para activar RAG, agrega `--rag <carpeta>` con archivos Markdown o texto. El recuperador local usa BM25 y no requiere una API externa ni un servicio de embeddings.

Ejemplo de pregunta: ¿Qué tipo de archivos acepta el ejemplo RAG? Respuesta: acepta documentos `.md` y `.txt` dentro de la carpeta indicada, incluidas sus subcarpetas.

Los fragmentos relevantes se agregan como contexto a la conversación. El agente debe responder con base en ellos, citar el archivo fuente y decir cuando la documentación no contiene la respuesta.
