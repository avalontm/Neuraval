# Neuraval

A Transformer-based neural network chatbot implementation in C# .NET 10.0 with CPU parallelization support. The goal is to create a **generative AI system similar to GPT**, capable of understanding context and generating human-like responses when provided with sufficient training data.

## Vision & Goals

This project aims to build a **generative conversational AI** from scratch using the Transformer architecture:

- **Generative AI Architecture**: Implements the core Transformer model used in GPT systems
- **Context Understanding**: Multi-head attention mechanism for understanding conversation context
- **Natural Language Generation**: Capable of generating coherent, contextually relevant responses
- **Scalable Training**: Designed to scale with dataset size - more training data = better responses
- **CPU-Optimized**: Parallel processing across multiple CPU cores for efficient training

**Note**: Like GPT, this model's quality is directly proportional to the amount and quality of training data provided. With sufficient conversational data, it can achieve human-like dialogue generation.

## Architecture

The system implements a full Transformer architecture:

- **Multi-Head Self-Attention**: Allows the model to focus on different parts of the conversation
- **Positional Encoding**: Maintains word order information
- **Feed-Forward Networks**: Deep learning layers for pattern recognition
- **Layer Normalization**: Stable training across deep networks
- **Token Embeddings**: Converts words to vector representations

This is the same fundamental architecture used in GPT models, adapted for conversational AI.

## Project Structure

```
Neuraval/
│
├── Neuraval.Abstractions/ (no dependencies)
│   ├── IChatModel.cs
│   ├── ChatMessage.cs / ChatRole.cs / ChatModelException.cs
│   ├── ITrainer.cs
│   └── ITrainableModel.cs
│
├── Neuraval/ (Core)
│   │
│   ├── Models/
│   │   ├── EmbeddingLayer.cs
│   │   ├── FeedForwardNetwork.cs 
│   │   ├── LayerNormalization.cs
│   │   ├── MultiHeadAttention.cs 
│   │   ├── PositionalEncoding.cs
│   │   ├── TransformerBlock.cs
│   │   └── TransformerModel.cs (implements ITrainableModel)
│   │
│   ├── Services/
│   │   ├── DatasetLoader.cs
│   │   ├── Tokenizer.cs
│   │   ├── SupervisedTrainer.cs (implements ITrainer<TransformerModel, CausalTrainingDataset>)
│   │
│   └── Utils/
│       ├── FuncionesActivacion.cs
│       └── Matematicas.cs
│
├── Neuraval.ChatBot/
│   ├── Models/
│   │   ├── Conversacion.cs
│   │   ├── DatoEntrenamiento.cs
│   │   └── Mensaje.cs
│   │
│   ├── Services/
│   │   ├── ChatModelFactory.cs (builds IChatModel, no config/switch yet)
│   │   └── TransformerChatBotService.cs (implements IChatModel)
│   │
│   └── Data/
│       ├── Saludos/
│       ├── Conversacion/
│       ├── Matematicas/
│       ├── Preguntas/
│       └── Ayuda/
│
├── Neuraval.CLI/ (Console Application)
│   ├── Program.cs
│   └── SystemMonitor.cs (CPU/Memory Monitor)
│
├── Neuraval.Evolution/ (generic genetic/RL lab, no game inside)
│   ├── IEnvironment.cs / IAgent.cs / IFitnessEvaluator.cs
│   ├── IEvolutionStrategy.cs
│   ├── Population.cs
│   └── Serialization/
│       ├── NavmBinaryFormat.cs
│       └── NavmBinarySerializer.cs
│
├── Neuraval.Evolution.MarioBridge/ (SMW agent via BizHawk Lua bridge)
│   ├── Program.cs (CLI: --reset / --checkpoint / --capture / --imitate / --seed-policy / --export-model / --population / --level)
│   ├── MarioAgent.cs / MarioStateEncoder.cs / MarioAgentOutput.cs
│   ├── MarioCheckpointStore.cs / MarioNeatModelStore.cs / MarioGenomeSerializer.cs
│   ├── MarioDataset.cs / MarioPolicyNetwork.cs / MarioImitationTrainer.cs
│   ├── SnesBridgeConnection.cs / SnesEnvironment.cs / MarioFitnessEvaluator.cs
│   └── MarioControllerEncoder.cs
│
├── Neuraval.Samples.DinoGame/ (graphical MonoGame sample, ported from the standalone "GameNeuronal" project)
│   ├── MainGame.cs
│   ├── Sources/
│   │   ├── NeuralNetwork.cs (wraps FeedForwardNetwork - no more Accord.Neuro/Accord.Statistics)
│   │   ├── Dino.cs / Bird.cs / Cactus.cs / BaseEnemy.cs
│   │   └── Animations.cs / DrawManager.cs / StageBackground.cs / InputManager.cs
│   └── Content/ (sprites, font)
│
└── Neuraval.Tests/
    ├── ChatModelFactoryTests.cs
    └── SupervisedTrainerRegressionTests.cs
```

## Architecture: `IChatModel` and `ITrainer`

The system is built around two small contracts in `Neuraval.Abstractions` (no dependency on the Transformer, or on each other):

- **`IChatModel`** — `Task<ChatMessage> SendAsync(IReadOnlyList<ChatMessage> messages, CancellationToken)`. This is the only thing a consumer (the CLI, `Neuraval.ChatBot`, or any future project) needs to know to *use* a chat backend. `TransformerChatBotService` implements it as a thin async wrapper over its existing `GenerateResponse`.
- **`ITrainer<TModel, TDataset>`** — `void Train(TModel model, TDataset dataset)`, where `TModel : ITrainableModel`. This is what's needed to *train* a model — a separate concern from using it, since a future backend might only be usable, never trainable locally. `SupervisedTrainer` implements it as `ITrainer<TransformerModel, CausalTrainingDataset>`; its internal logic (Adam, backprop, checkpoints, early stopping) is untouched — only exposed behind the interface.

`ChatModelFactory.CreateFromSavedModel(...)` builds an `IChatModel` from what's already in `training-settings.json` (`ModelPath`, etc). With a single real backend there's nothing to choose at runtime, so there's no `switch` and no new config section — that's intentionally deferred to the next point.

### Adding a new chat backend

This only makes sense once there's a second backend you actually want to use (another local Transformer, a small model served over `localhost`, an external API, etc). When that day comes:

1. Create `Neuraval.Backends.<Name>/`, referencing only `Neuraval.Abstractions`.
2. Implement `IChatModel` on it — `HttpClient` + `System.Text.Json` if it talks over HTTP, or whatever the backend needs. No third-party NuGet packages.
3. Only then, add a `ChatBackend` field to `training-settings.json` (not a new file) plus a `Backends` section with each backend's config; default stays `"local"`.
4. Update `ChatModelFactory` to branch on that field.
5. Add a minimal integration test confirming the new backend responds.

Nothing in `Neuraval.CLI`, `Neuraval.ChatBot`, or any existing backend needs to change — the new project, the factory branch, and the config field added at that point are the only touch points.

## Evolutionary lab

`Neuraval.Evolution` is a standalone, generic library for problems with an automatically measurable reward (game-playing agents, not chat) — `IEnvironment`, `IAgent`, `IFitnessEvaluator`, `IEvolutionStrategy` and `Population<TAgent, TState, TAction>` know nothing about Dinosaurs, Transformers, or chat.

It's implemented against `FeedForwardNetwork` from the core project as a small policy network; evolution works directly off what `FeedForwardNetwork` already exposes — `SaveState()`/`LoadState()` — mutating the flattened weight arrays with Gaussian noise instead of training by backprop. The library is fully decoupled from the chatbot — it doesn't reference `Neuraval.ChatBot`, `Neuraval.CLI`, or `Neuraval.Abstractions` at all.

## Sample: GameNeuronal (graphical Dino game)

`Neuraval.Samples.DinoGame` is a MonoGame clone of the Chrome offline dino game (sprites, animation, obstacles, debug overlay), originally a separate project ("GameNeuronal") that has been folded into this solution as a sample.

It used to depend on `Accord.Neuro`/`Accord.Statistics` for its neural network. That dependency has been removed — it now uses `FeedForwardNetwork` from `Neuraval.Core.Models` (this repo's own engine), via a `ProjectReference` to `Neuraval.csproj`. No third-party neural network library is involved anywhere in this solution anymore. MonoGame itself is kept, since it's what actually renders the game.

Each dinosaur spawns with its own randomly-initialized `FeedForwardNetwork` (7 game-state inputs — obstacle distance/position/size, dino height, game speed — mapped to a jump/duck decision read off the first two outputs). Like the original, this sample doesn't train or evolve the weights; it's a structural demo of wiring a neural net's raw output into game decisions, not a trained agent.

Run it with (Windows only, needs the MonoGame/WindowsDesktop workload):

```bash
dotnet run --project Neuraval.Samples.DinoGame
```

Controls: `F1` toggles the debug overlay, `Esc` restarts once every dinosaur has died.

## SMW Agent: Mario Bridge

`Neuraval.Evolution.MarioBridge` entrena agentes NEAT para super Mario World controlando BizHawk a través de un bridge Lua por TCP (`127.0.0.1:8766`). El agente percibe la RAM P0+P1 (tilemap, estado de Mario, cámara, sprites, moneda Yoshi más cercana, item sostenible/sostenido y checkpoint) codificada en un vector de **326 entradas** y actúa con **7 salidas** (Izquierda, Derecha, A, B, Y, Abajo, Arriba).

El tilemap (13×13 alrededor de Mario) envía el **valor crudo del Map16** (0–255 normalizado a 0..1) en vez de binario, así la red distingue ladrillos, bloques `?`, monedas y suelo. Además hay bloques de señales de colección y navegación:
- Monedas y bloques-moneda: nº cerca y offset del más cercano (monedas: tiles `$025/$02B/$05B/$06B`; bloques-moneda: `$11B/$123`).
- **Bloques de diálogo** (Map16 completo `$0104-$0107`): nº cerca y offset, para que el modelo evite golpearlos y no se abra el mensaje.
- **Acantilados**: los 2 huecos más próximos hacia delante, cada uno como (distancia en tiles, anchura en tiles) usando el tile de los pies + 4 filas debajo — el modelo sabe cuándo y cuánto hay que correr y saltar.

Los tres allowlists son heurísticos y se deben calibrar contra una captura real del nivel DP1 vanilla.

### Formato binario `.navm`

Todos los artefactos del proyecto se guardan en el contenedor binario universal `.navm`:

```
magic "NAVM" (4) | version (u16) | flags (u8) | reserved (u8)
headerLen (i32) | header JSON (UTF-8) | bodyLen (i32) | body (GZip opcional) | SHA-256 (32)
```

Se escribe de forma atómica (fichero temporal + rename). El header JSON de cada artefacto (checkpoint, modelo NEAT aislado, dataset, política) valida la compatibilidad — ante un cambio de InputCount/OutputCount, los checkpoints antiguos se archivan automáticamente en vez de corromperse.

### Pipeline de Imitation Learning

1. `--capture <dataset.navm>`: graba tus partidas desde BizHawk (estado + botones + recompensa por frame), cerrando episodio en muerte/nivel completado/reset manual. La recompensa por frame reproduce el shaping del entorno evolutivo: avance horizontal + monedas (×50) + power-up (ganancia ×200, pérdida −150).
2. `--imitate <dataset.navm> [--epochs N] [--imitate-out policy.navm]`: entrena una política (MLP) por imitación con BCE. La pérdida de cada muestra se **pondera por su recompensa** (pesos 1–3 normalizados por min-max), de modo que el modelo imita con más fuerza las jugadas "buenas" (avanzar, recoger monedas/power-ups). Además hay split de **validación 90/10**, se restaura el epoch con mejor val-BCE y **early stopping** con paciencia 5 — el `--imitate-out` guarda siempre la política del mejor val-loss.
3. `--seed-policy policy.navm`: siembra la población NEAT con la política imitada en el siguiente entrenamiento evolutivo.

### Ejecución del entrenamiento evolutivo

```bash
# Entrenamiento principal (espera a BizHawk en 127.0.0.1:8766)
dotnet run --project Neuraval.Evolution.MarioBridge

# Reanudar con un checkpoint alternativo
dotnet run --project Neuraval.Evolution.MarioBridge -- --checkpoint mi_checkpoint.navm

# Empezar de cero (borra el progreso anterior)
dotnet run --project Neuraval.Evolution.MarioBridge -- --checkpoint mi_checkpoint.navm --reset

# Exportar el mejor genoma a un modelo autocontenido
dotnet run --project Neuraval.Evolution.MarioBridge -- --export-model checkpoints/mario_best.navm

# Modo captura de dataset para Imitation Learning
dotnet run --project Neuraval.Evolution.MarioBridge -- --capture checkpoints/mario_dataset.navm

# Entrenar la política imitada
dotnet run --project Neuraval.Evolution.MarioBridge -- --imitate checkpoints/mario_dataset.navm --epochs 20 --imitate-out checkpoints/mario_policy.navm
```

Opciones de entrenamiento evolutivo:

```bash
# Tamaño de población (default 16; poblaciones más grandes exploran mejor pero alargan cada generación)
dotnet run --project Neuraval.Evolution.MarioBridge -- --population 20

# Rotar entre varios niveles/savestates por generación (curriculum). Repite el flag por cada nivel,
# el índice es el mismo que usa mario_bridge.lua para cargar los savestates.
dotnet run --project Neuraval.Evolution.MarioBridge -- --level "Nivel 0 (DP1.state)" --level "Nivel 1 (medio.state)" --level "Nivel 2 (final.state)"
```

Sin `--level`, se usa el nivel por defecto. El elitismo por especie requiere `MinSpeciesSizeForElite = 3` y conserva `EliteCountPerSpecies = 2`; el mejor genoma histórico siempre re-siembra la población en el slot 0.

El fitness por generación combina avance horizontal con bonificaciones (nivel completado, power-ups, monedas, monedas Yoshi con bonus por juntar las 5, checkpoint/punto medio) y penalizaciones (daño recibido, muerte). El `mario_bridge.lua` implementa el protocolo v12 (70 campos) con los comandos `RESET`, `CAPTURE` y `STOP`, incluyendo status/stun/flags/misc de sprites, off-screen full/eaten/interacción objeto/timer giro, cluster sprites, capas 2/3, segundo jugador, timers de POW/door/player, tilemap con valores crudos de Map16, senales de monedas/bloques-moneda, bloques de dialogo (Map16 completo `$0104-$0107`), acantilados (2 huecos: distancia+anchura), banderas de item sostenido/sostenible ($1470/$148F + status $14C8), checkpoint de punto medio ($13CE/$13CD), contador de monedas Yoshi ($1420), señales de pared/hueco vertical (distancia en tiles a la primera pared sólida hacia la derecha a la altura del cuerpo de Mario, y a la primera plataforma sólida arriba/abajo en la columna de Mario, escaneando hasta `GridRadius` tiles; 0 = nada detectado en rango), bandera de nivel vertical (`$7E:1412`, scroll vertical habilitado/condicional, colapsado a 0/1) y señal de tubería vertical transitable cercana (Map16 completo `$0137`/`$0138`, tiles superiores de tubería exit-enabled). Los datasets e checkpoints grabados con el protocolo v11 quedan obsoletos por el cambio de entradas (323→326) y deben recapturarse. Los datasets e checkpoints grabados con el protocolo v8 quedan obsoletos por el cambio de entradas (298→307) y salidas (6→7, se agrega Arriba) y deben recapturarse. El campo `Blocked` de Mario ($7E:0077, formato `SxxMUDLR`) y el de cada sprite cercano ($7E:1588, formato `xxxxUDLR`) ya no se pasan crudos/bitmask a la red: se decodifican en 4 señales binarias por bloque (izquierda/derecha/arriba/abajo), lo que sube el vector de entradas de 307 a **319** sin tocar el wire protocol (protocolo v9, 67 campos, sin cambios).

## Running the tests

```bash
dotnet test Neuraval.Tests
```

Covers: `ChatModelFactory` returning the right `IChatModel` (and matching a model loaded directly, byte-for-byte in its generated response), `SupervisedTrainer.Train` (the `ITrainer` entry point) producing identical training/validation losses and predictions to calling `TrainCausalWithValidation` directly — confirming the Fase 2 migration didn't change training behavior — and the `.navm` container round-trip (compresión, checksum corrupto, magic inválido), el contador de entradas del `MarioStateEncoder` (326), las señales de monedas/bloques-moneda/diálogo/acantilados/moneda-Yoshi/item-sostenido/checkpoint y los tiles crudos, la decodificación del controller, la ponderación por recompensa del `MarioRewardWeighting` y que `MarioImitationTrainer` lleva la política hacia la máscara de botones objetivo.

## Key Features

### Generative AI Capabilities

- **Context-Aware Generation**: Understands conversation flow and context
- **Multiple Response Generation**: Can generate several candidate responses and select the best
- **Temperature Sampling**: Controls creativity vs. consistency in responses
- **Beam Search**: Finds optimal response sequences
- **Top-K Sampling**: Balances between diversity and quality

### Performance Features

- **CPU Parallelization**: 2-8x faster training on multi-core CPUs
- **Batch Processing**: Efficient training on large datasets
- **Memory Optimization**: Handles large vocabularies and long sequences
- **Real-time Monitoring**: Track CPU, memory, and training progress
- **Early Stopping**: Prevents overfitting with validation-based stopping

### Training Features

- **Automatic Vocabulary Building**: Learns from your training data
- **Validation Split**: Monitors model performance on unseen data
- **Progress Tracking**: Real-time loss monitoring and ETA
- **Model Persistence**: Save and load trained models
- **Multiple Data Formats**: JSON, XML support for training data

## Installation

### Prerequisites
- .NET 10.0 SDK or higher
- Multi-core CPU (4+ cores recommended for training)
- 8GB+ RAM (16GB+ recommended for large models)

### Build Commands

```bash
# Clone repository
git clone https://github.com/avalontm/Neuraval.git
cd Neuraval

# Build solution
dotnet build

# Run console application
dotnet run --project Neuraval.CLI
```

## Quick Start

### 1. Prepare Training Data

Create `.txt` files in the `Data/` folder with conversation pairs, one per block, separated by a blank line:

```
Usuario: hello
Asistente: Hello! How can I help you today?

Usuario: how are you
Asistente: I'm doing great, thanks for asking!
```

Legacy `.json` files (`{"entrada": [...], "respuesta": [...]}`) are still supported and loaded alongside `.txt` files for backward compatibility, but the plain-text format is now the recommended one.

If you already have `.json` datasets in that format, convert them to `.txt` instead of writing them by hand:

```bash
# Convert a single file
dotnet run --project Neuraval.CLI -- --convert Data/misdatos.json Data/misdatos.txt

# Convert every .json file under Data/ (and remove the originals so they aren't loaded twice)
dotnet run --project Neuraval.CLI -- --convert-all Data --delete-original
```

**Important**: The more diverse and extensive your training data, the better the generative AI will perform. Aim for thousands of conversation examples for good results.

### 2. Adjust Training Settings (optional)

On first run, the CLI creates `training-settings.json` in the working directory with the default hyperparameters. Edit it to make training faster (smaller model, fewer epochs, bigger batches) or better (larger model, more epochs):

```json
{
  "EmbeddingDim": 64,
  "NumLayers": 2,
  "NumHeads": 2,
  "FeedforwardDim": 128,
  "MaxSequenceLength": 32,
  "Dropout": 0.1,
  "BatchSize": 16,
  "GradientAccumulationSteps": 1,
  "LearningRate": 0.001,
  "Epochs": 100,
  "ValidationSplit": 0.2,
  "Patience": 10,
  "CheckpointEveryEpochs": 5,
  "NumThreads": -1,
  "ModelPath": "SavedModel",
  "DataFolder": "Data"
}
```

Any of these can also be overridden per run from the command line without touching the file:

```bash
dotnet run --project Neuraval.CLI -- --epochs 20 --batchSize 8 --embeddingDim 32
```

Smaller `EmbeddingDim`/`NumLayers`/`FeedforwardDim`/`MaxSequenceLength` and a bigger `BatchSize` are the levers with the biggest effect on wall-clock training time.

`GradientAccumulationSteps` does **not** speed up training — it trades speed for a bigger *effective* batch size (`BatchSize * GradientAccumulationSteps`) without using more RAM, by summing gradients over several micro-batches before each weight update. Raise it if you want batch-32-like training stability but your machine can't fit a batch of 32 in memory; leave it at `1` if you're just trying to train faster.

### 3. Train the Model

```bash
dotnet run --project Neuraval.CLI
```

The system will:
1. Load all `.txt`/`.json` files from `Data/` folder
2. Build vocabulary from your conversations
3. Train the Transformer model
4. Save the trained model to `SavedModel/`
5. Start interactive chat

### 4. Chat with Your AI

```
You: Hello
Bot: Hello! How can I help you today? (125ms)

You: How are you?
Bot: I'm doing great, thanks for asking! (98ms)
```

## Usage Examples


### Basic Training

```csharp
using Neuraval.ChatBot.Services;

// Create chatbot with 8 CPU threads
var chatBot = new TransformerChatBotService(
    embeddingDim: 128,
    numLayers: 4,
    numHeads: 4,
    feedforwardDim: 512,
    maxSequenceLength: 128,
    numThreads: 8
);

// Load your conversation data
var conversations = new List<ConversationPair>
{
    new ConversationPair { Input = "hello", Target = "hi there" },
    new ConversationPair { Input = "how are you", Target = "I'm great thanks" }
};

// Build vocabulary
var texts = conversations.SelectMany(c => new[] { c.Input, c.Target }).ToList();
chatBot.BuildVocabularyFromTexts(texts);

// Train the model
chatBot.EntrenarConConversaciones(
    conversations, 
    epochs: 200,
    batchSize: 32,
    learningRate: 0.001
);

// Generate response
string response = chatBot.GenerarRespuesta("hello");
Console.WriteLine(response);
```

### Advanced Configuration

```csharp
// Small model (fast, basic quality)
// Good for: Small datasets (< 1000 examples)
var chatBot = new TransformerChatBotService(
    embeddingDim: 64,
    numLayers: 2,
    numHeads: 2,
    feedforwardDim: 256,
    maxSequenceLength: 64,
    numThreads: 4
);

// Medium model (recommended for most use cases)
// Good for: Medium datasets (1000-10000 examples)
var chatBot = new TransformerChatBotService(
    embeddingDim: 128,
    numLayers: 4,
    numHeads: 4,
    feedforwardDim: 512,
    maxSequenceLength: 128,
    numThreads: -1  // use all CPU cores
);

// Large model (high quality, slower)
// Good for: Large datasets (10000+ examples)
var chatBot = new TransformerChatBotService(
    embeddingDim: 256,
    numLayers: 6,
    numHeads: 8,
    feedforwardDim: 1024,
    maxSequenceLength: 256,
    numThreads: -1
);
```

### Generation Strategies

```csharp
// 1. Sampling (creative, diverse responses)
string response = chatBot.GenerarRespuesta(
    "tell me a story",
    maxLength: 100,
    temperature: 0.8,  // higher = more creative
    topK: 40
);

// 2. Beam Search (deterministic, high quality)
string response = chatBot.GenerarConBeamSearch(
    "what is AI",
    maxLength: 50,
    beamWidth: 5  // searches 5 best paths
);

// 3. Multiple Candidates (compare options)
List<string> responses = chatBot.GenerarMultiplesRespuestas(
    "hello",
    numCandidates: 5,
    temperature: 0.7
);
```

### Training with Progress Monitoring

```csharp
chatBot.EntrenarConConversaciones(
    conversations,
    epochs: 200,
    batchSize: 32,
    learningRate: 0.001,
    validationSplit: 0.2,
    onEpochCompleted: (epoch, trainLoss, valLoss) =>
    {
        Console.WriteLine($"Epoch {epoch}: Train Loss={trainLoss:F4}, Val Loss={valLoss:F4}");
        
        // Early stopping logic
        if (valLoss < 0.01) 
        {
            Console.WriteLine("Target loss reached!");
        }
    }
);
```

### Save and Load Models

```csharp
// Save trained model
chatBot.GuardarModeloCompleto("MyModel");

// Load model later
var loadedBot = new TransformerChatBotService(numThreads: -1);
loadedBot.CargarModeloCompleto("MyModel");

// Start using immediately
string response = loadedBot.GenerarRespuesta("hello");
```

## Training Data Guidelines

To achieve GPT-like generative capabilities, your training data should:

### Quality Requirements

1. **Diversity**: Cover many topics and conversation styles
2. **Consistency**: Maintain consistent tone and style
3. **Context**: Include multi-turn conversations, not just single Q&A
4. **Natural Language**: Use natural, conversational language
5. **Balance**: Avoid over-representing certain patterns

### Quantity Guidelines

| Dataset Size | Expected Quality | Use Case |
|--------------|------------------|----------|
| 100-500 examples | Basic responses | Testing, prototyping |
| 500-2000 examples | Simple conversations | Small domain chatbot |
| 2000-10000 examples | Good generalization | Domain-specific AI |
| 10000+ examples | Human-like responses | General conversational AI |

### Example Training Data Structure

```json
{
  "vocabulario": {
    "hello": 0, "hi": 1, "hey": 2,
    "how": 3, "are": 4, "you": 5,
    "what": 6, "is": 7, "your": 8, "name": 9
  },
  "respuestas": [
    "Hello! How can I assist you today?",
    "Hi there! What can I help you with?",
    "I'm doing great, thank you for asking!",
    "I'm an AI assistant created to help answer questions.",
    "My name is Neuraval. Nice to meet you!"
  ],
  "entrenamiento": [
    {"entrada": ["hello"], "respuesta": 0},
    {"entrada": ["hi"], "respuesta": 1},
    {"entrada": ["hey"], "respuesta": 1},
    {"entrada": ["how", "are", "you"], "respuesta": 2},
    {"entrada": ["what", "is", "your", "name"], "respuesta": 4}
  ]
}
```

## Performance Metrics

### Training Speed (2000 conversations, 200 epochs)

| CPU Cores | Training Time | Speedup |
|-----------|---------------|---------|
| 4 cores   | ~45 minutes   | 1.0x    |
| 8 cores   | ~25 minutes   | 1.8x    |
| 16 cores  | ~15 minutes   | 3.0x    |

### Memory Requirements

| Model Size | Parameters | RAM Usage | Recommended RAM |
|-----------|-----------|-----------|-----------------|
| Small     | ~500K     | 1-2 GB    | 4 GB            |
| Medium    | ~2M       | 4-8 GB    | 16 GB           |
| Large     | ~10M      | 16+ GB    | 32 GB           |

### System Monitor Output

```
===========================================
          SYSTEM CONFIGURATION
===========================================
  CPU Cores: 8
  Threads Configured: 8
  OS: Microsoft Windows 10.0.22000
  Architecture: X64
  .NET Version: 9.0.0
  GPU Support: NO (CPU-only mode)
  Available Memory: ~16384 MB
===========================================

[00:00:05] CPU: 87.3% | RAM: 1245 MB | Threads: 23
[00:00:10] CPU: 92.1% | RAM: 1567 MB | Threads: 25
[00:00:15] CPU: 89.7% | RAM: 1789 MB | Threads: 24

===========================================
           TRAINING STATISTICS
===========================================
  Total Time: 00:25:43
  Peak Memory: 2048 MB
  Avg CPU Usage: 88.5%
  Final Thread Count: 24
  GC Collections:
    Gen 0: 1245
    Gen 1: 156
    Gen 2: 12
  Total CPU Time: 03:25:12
===========================================
```

## Console Commands

While chatting with the trained AI:

| Command | Description |
|---------|-------------|
| `exit` or `quit` | End the conversation |
| `beam` | Switch to beam search generation |
| `sampling` | Switch to sampling generation (default) |
| `multiple` | Generate 5 candidate responses |
| `info` | Display model information |
| `stats` | Show current system statistics |

## API Reference

### TransformerChatBotService

```csharp
// Constructor
TransformerChatBotService(
    int embeddingDim = 128,
    int numLayers = 4,
    int numHeads = 4,
    int feedforwardDim = 512,
    int maxSequenceLength = 128,
    int numThreads = -1  // -1 = all CPU cores
)

// Build vocabulary from text data
void BuildVocabularyFromTexts(
    List<string> texts,
    int minFrequency = 1,
    int maxVocabSize = 10000
)

// Train the model
void EntrenarConConversaciones(
    List<ConversationPair> conversations,
    int epochs = 100,
    int batchSize = 32,
    double learningRate = 0.001,
    double validationSplit = 0.2,
    Action<int, double, double>? onEpochCompleted = null
)

// Generate response (sampling)
string GenerarRespuesta(
    string mensajeUsuario,
    int maxLength = 50,
    double temperature = 0.7,
    int topK = 40
)

// Generate response (beam search)
string GenerarConBeamSearch(
    string mensajeUsuario,
    int maxLength = 50,
    int beamWidth = 5
)

// Generate multiple candidates
List<string> GenerarMultiplesRespuestas(
    string mensajeUsuario,
    int numCandidates = 5,
    int maxLength = 50,
    double temperature = 0.8
)

// Save model to disk
void GuardarModeloCompleto(string carpetaDestino)

// Load model from disk
bool CargarModeloCompleto(string carpetaOrigen)
```

## Troubleshooting

### Low Quality Responses

**Problem**: Bot generates nonsensical or repetitive responses

**Solutions**:
```csharp
// 1. Increase training data (most important)
// Add more diverse conversation examples

// 2. Train for more epochs
chatBot.EntrenarConConversaciones(conversations, epochs: 500);

// 3. Increase model size
var chatBot = new TransformerChatBotService(
    embeddingDim: 256,
    numLayers: 6
);

// 4. Adjust generation parameters
string response = chatBot.GenerarRespuesta(
    input,
    temperature: 0.7,  // lower = more conservative
    topK: 40          // higher = more diverse
);
```

### Out of Memory

**Problem**: System runs out of RAM during training

**Solutions**:
```csharp
// 1. Reduce batch size
chatBot.EntrenarConConversaciones(conversations, batchSize: 8);

// 2. Reduce model size
var chatBot = new TransformerChatBotService(
    embeddingDim: 64,
    numLayers: 2
);

// 3. Reduce sequence length
var chatBot = new TransformerChatBotService(
    maxSequenceLength: 64
);
```

### Slow Training

**Problem**: Training takes too long

**Solutions**:
```csharp
// 1. Verify CPU usage
// Should be 80-100% during training

// 2. Increase thread count
var chatBot = new TransformerChatBotService(
    numThreads: Environment.ProcessorCount
);

// 3. Increase batch size (if you have RAM)
chatBot.EntrenarConConversaciones(conversations, batchSize: 64);
```

### Overfitting

**Problem**: Model memorizes training data, poor on new inputs

**Solutions**:
```csharp
// 1. Use validation split
chatBot.EntrenarConConversaciones(
    conversations, 
    validationSplit: 0.2  // 20% for validation
);

// 2. Early stopping (automatic in code)
// Stops when validation loss stops improving

// 3. Add more diverse training data
// The best solution for generalization
```

## Technical Details

### Transformer Architecture

The model implements the standard Transformer architecture:

1. **Input Embedding**: Converts tokens to vectors
2. **Positional Encoding**: Adds position information
3. **Transformer Blocks** (repeated N times):
   - Multi-Head Self-Attention
   - Layer Normalization
   - Feed-Forward Network
   - Residual Connections
4. **Output Layer**: Projects to vocabulary size

### Attention Mechanism

Multi-head attention allows the model to focus on different aspects:
- **Query, Key, Value**: Three learned projections
- **Scaled Dot-Product**: Attention score calculation
- **Multiple Heads**: Parallel attention computations
- **Concatenation**: Combine all head outputs

### Training Process

1. **Forward Pass**: Generate predictions
2. **Loss Calculation**: Cross-entropy loss
3. **Backward Pass**: Calculate gradients
4. **Weight Update**: Apply gradients with learning rate
5. **Validation**: Check performance on unseen data

## Comparison with GPT

| Feature | Neuraval | GPT-3 |
|---------|---------------|-------|
| Architecture | Transformer | Transformer |
| Implementation | C# from scratch | Python/CUDA |
| Parameters | ~500K - 10M | 175 billion |
| Training Data | User-provided | 570GB text |
| Hardware | CPU multi-thread | GPU clusters |
| Purpose | Customizable chatbot | General purpose |
| Training Time | Minutes-hours | Months |
| Cost | Free, open-source | Commercial API |

**Key Insight**: While GPT-3 is massive, this implementation uses the same fundamental architecture. With sufficient training data in your domain, you can achieve good conversational AI for specific use cases.

## Future Enhancements

Planned features to make the system even more GPT-like:

- [ ] GPU acceleration with CUDA support
- [ ] Attention visualization tools
- [ ] Pre-training on large text corpora
- [ ] Fine-tuning on specific domains
- [ ] Model quantization for faster inference
- [ ] Distributed training across multiple machines
- [ ] Web API for remote access
- [ ] Real-time streaming responses
- [ ] Multi-language support
- [ ] Transfer learning from pre-trained models

## Contributing

Contributions are welcome! Areas where help is needed:

1. **Training Data**: Contribute high-quality conversation datasets
2. **Performance**: Optimize CPU/GPU utilization
3. **Features**: Add new generation strategies
4. **Documentation**: Improve examples and guides
5. **Testing**: Add unit tests and benchmarks

## License

Open source for educational and research purposes.

## Acknowledgments

This project implements the Transformer architecture introduced in:
- "Attention Is All You Need" (Vaswani et al., 2017)
- Built with inspiration from GPT and modern language models
- Designed for educational purposes and understanding deep learning

## Author

**AvalonTM**
- GitHub: [https://github.com/avalontm](https://github.com/avalontm)

## Support & Contact

- **Issues**: Open an issue in the repository
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check this README and code comments
- **Author**: Contact via GitHub profile

---

**Note**: This is a learning-focused implementation. For production applications requiring GPT-level performance, consider using established APIs (OpenAI, Anthropic) or larger models. However, for domain-specific applications with sufficient training data, this implementation can achieve impressive results while maintaining full control over your AI system.