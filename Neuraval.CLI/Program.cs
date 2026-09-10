using Neuraval.Abstractions;
using Neuraval.ChatBot.Services;
using Neuraval.Core.Models;
using Neuraval.Core.Quantization;
using Neuraval.Core.Serialization;
using Neuraval.Core.Services;
using Neuraval.Core.Utils;
using Neuraval.Cuda;
using Neuraval.Tensor;
using System.Diagnostics;
using System.Linq;

namespace Neuraval.CLI
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            bool forceContinueTraining = args.Any(a => string.Equals(a, "--continue", StringComparison.OrdinalIgnoreCase));

            if (args.Any(a => string.Equals(a, "--int8-inference", StringComparison.OrdinalIgnoreCase)))
            {
                Int8InferenceSettings.Enable();
                Console.WriteLine("Inferencia INT8 en CPU habilitada (Fase 5.4.2, experimental — medí con --int8-benchmark).");
            }

            if (args.Length > 0 && args[0] == "--benchmark")
            {
                RunTrainingBenchmark();
                return;
            }

            if (args.Length > 0 && args[0] == "--gpu-benchmark")
            {
                RunGpuBenchmark();
                return;
            }

            if (args.Length > 0 && args[0] == "--benchmark-suite")
            {
                RunBenchmarkSuite();
                return;
            }

            if (args.Length > 0 && args[0] == "--memory-profile")
            {
                RunMemoryProfile();
                return;
            }

            if (args.Length > 0 && args[0] == "--convert")
            {
                RunConvert(args);
                return;
            }

            if (args.Length > 0 && args[0] == "--convert-all")
            {
                RunConvertAll(args);
                return;
            }

            if (args.Length > 0 && args[0] == "--quantize")
            {
                RunQuantize(args);
                return;
            }

            if (args.Length > 0 && args[0] == "--export-lora")
            {
                RunExportLora(args);
                return;
            }

            if (args.Length > 0 && args[0] == "--import-lora")
            {
                RunImportLora(args);
                return;
            }

            if (args.Length > 0 && args[0] == "--int8-benchmark")
            {
                RunInt8Benchmark(args);
                return;
            }

            if (args.Length > 0 && args[0] == "--chat")
            {
                RunChatOnly(args);
                return;
            }

            Console.WriteLine("===========================================");
            Console.WriteLine("       Neuraval — Entrenamiento");
            Console.WriteLine("===========================================");
            Console.WriteLine();

            var settings = TrainingSettings.Load("training-settings.json", args);
            settings.Print();

            if (settings.UseGpu)
            {
                bool gpuReady = Matematicas.TryEnableGpu();
                Console.WriteLine(gpuReady
                    ? "GPU CUDA detectada y activada para operaciones soportadas."
                    : "No se pudo activar la GPU CUDA, se continúa en CPU.");
                Console.WriteLine();
            }

            string modelPath = settings.ModelPath;
            string logFilePath = Path.Combine(modelPath, "training_log.csv");
            int checkpointEveryEpochs = settings.CheckpointEveryEpochs;

            int numThreads = settings.NumThreads > 0 ? settings.NumThreads : Environment.ProcessorCount;

            var chatBot = new TransformerChatBotService(
                embeddingDim: settings.EmbeddingDim,
                numLayers: settings.NumLayers,
                numHeads: settings.NumHeads,
                feedforwardDim: settings.FeedforwardDim,
                maxSequenceLength: settings.MaxSequenceLength,
                dropout: settings.Dropout,
                numThreads: numThreads
            );

            bool modelLoaded = false;
            TrainingProgressState? resumeProgress = null;

            if (Directory.Exists(modelPath))
            {
                Console.WriteLine("Se encontró un modelo guardado. Cargando...");
                Console.WriteLine();

                modelLoaded = chatBot.LoadCompleteModel(modelPath);

                if (modelLoaded && chatBot.IsTrained() && !forceContinueTraining)
                {
                    Console.WriteLine("¡Modelo cargado correctamente!");
                    Console.WriteLine();
                    Console.WriteLine("(El modelo ya completó su entrenamiento. Usa --continue para seguir entrenándolo con los datos actuales de la carpeta de datos.)");
                    Console.WriteLine();
                    StartChat(chatBot).GetAwaiter().GetResult();
                    return;
                }

                if (modelLoaded)
                {
                    resumeProgress = TransformerChatBotService.LoadTrainingProgress(modelPath);

                    if (resumeProgress != null)
                    {
                        Console.WriteLine($"Se encontró un entrenamiento interrumpido en la época {resumeProgress.LastCompletedEpoch}. Continuando el entrenamiento...");
                        Console.WriteLine();
                    }
                    else if (chatBot.IsTrained() && forceContinueTraining)
                    {
                        Console.WriteLine("El modelo ya había completado su entrenamiento. Se continuará entrenando desde los pesos actuales con los datos disponibles.");
                        Console.WriteLine();
                    }
                }
                else
                {
                    Console.WriteLine("No se pudo cargar el modelo. Se entrenará uno nuevo...");
                    Console.WriteLine();
                }
            }

            Console.WriteLine($"Buscando datos de entrenamiento en la carpeta '{settings.DataFolder}'...");
            Console.WriteLine();

            try
            {
                var conversations = LoadTrainingData(settings.DataFolder);

                if (conversations == null || conversations.Count == 0)
                {
                    Console.WriteLine("Error: no se encontraron datos de entrenamiento.");
                    Console.WriteLine();
                    Console.WriteLine($"Agrega un archivo .txt a la carpeta '{settings.DataFolder}' con este formato:");
                    Console.WriteLine();
                    Console.WriteLine(@"Usuario: hola como estas
Asistente: muy bien gracias y tu

Usuario: cuanto es dos mas dos
Asistente: el resultado es cuatro");
                    Console.WriteLine();
                    Console.WriteLine("¿Tienes datos en el formato JSON indexado antiguo (entrada/respuesta)?");
                    Console.WriteLine("Conviértelos primero a texto plano con:");
                    Console.WriteLine("  dotnet run --project Neuraval.CLI -- --convert-all " + settings.DataFolder);
                    return;
                }

                Console.WriteLine($"Total de conversaciones cargadas: {conversations.Count}");
                Console.WriteLine();

                if (!modelLoaded)
                {
                    if (conversations.Count < 50)
                    {
                        Console.WriteLine("ADVERTENCIA: el dataset es muy pequeño.");
                        Console.WriteLine("Para mejores resultados se recomiendan al menos 500-1000 pares de conversación.");
                        Console.WriteLine("Con tan pocos ejemplos el modelo puede no aprender bien.");
                        Console.WriteLine();
                    }

                    var allTexts = conversations
                        .SelectMany(c => new[] { c.Input, c.Target })
                        .ToList();

                    Console.WriteLine("Construyendo vocabulario...");
                    chatBot.BuildVocabularyFromTexts(allTexts, minFrequency: 1, maxVocabSize: settings.VocabSize);
                    Console.WriteLine($"Tamaño del vocabulario: {chatBot.GetVocabularySize()}");
                    Console.WriteLine();
                }
                else
                {
                    Console.WriteLine($"Continuando con el vocabulario existente del checkpoint (tamaño: {chatBot.GetVocabularySize()}).");
                    Console.WriteLine();
                }

                ApplyLoraSettingsIfRequested(chatBot, settings);

                using (var monitor = new SystemMonitor())
                {
                    Console.WriteLine("Entrenando el modelo Transformer...");
                    Console.WriteLine("Configuración:");
                    Console.WriteLine($"  Ejemplos totales: {conversations.Count}");
                    Console.WriteLine($"  Tamaño de vocabulario: {chatBot.GetVocabularySize()}");
                    Console.WriteLine($"  Tamaño del modelo: {settings.EmbeddingDim} dim, {settings.NumLayers} capas, {settings.NumHeads} cabezas");
                    Console.WriteLine($"  Estrategia: early stopping con paciencia {settings.Patience}");
                    Console.WriteLine();

                    int totalEpochs = settings.Epochs;
                    int startEpoch = resumeProgress != null ? resumeProgress.LastCompletedEpoch + 1 : 0;
                    var startTime = DateTime.Now;
                    int callbackCount = 0;
                    double bestLoss = resumeProgress?.BestValidationLoss ?? double.MaxValue;
                    int epochsSinceBest = resumeProgress?.EpochsWithoutImprovement ?? 0;

                    chatBot.TrainWithConversations(
                        conversations,
                        epochs: totalEpochs,
                        batchSize: settings.BatchSize,
                        gradientAccumulationSteps: settings.GradientAccumulationSteps,
                        learningRate: settings.LearningRate,
                        validationSplit: settings.ValidationSplit,
                        patience: settings.Patience,
                        checkpointFolder: modelPath,
                        checkpointEveryEpochs: checkpointEveryEpochs,
                        logFilePath: logFilePath,
                        resumeFrom: resumeProgress,
                        onEpochCompleted: (epoch, trainLoss, valLoss) =>
                        {
                            callbackCount++;

                            if (valLoss < bestLoss)
                            {
                                bestLoss = valLoss;
                                epochsSinceBest = 0;
                            }
                            else
                            {
                                epochsSinceBest++;
                            }

                            double progress = (double)(epoch + 1) / totalEpochs * 100;
                            var elapsed = DateTime.Now - startTime;
                            int epochsProcessedThisRun = epoch - startEpoch + 1;
                            var estimatedTotalThisRun = TimeSpan.FromSeconds(elapsed.TotalSeconds / epochsProcessedThisRun * (totalEpochs - startEpoch));
                            var remaining = estimatedTotalThisRun - elapsed;

                            int barLength = 40;
                            int filled = (int)(progress / 100 * barLength);
                            string bar = new string('=', filled) + new string('-', barLength - filled);

                            Console.Write($"\r[{bar}] {progress:F1}% | Época {epoch + 1}/{totalEpochs} | " +
                                        $"Train: {trainLoss:F6} | Val: {valLoss:F6} | Mejor: {bestLoss:F6} | " +
                                        $"Paciencia: {epochsSinceBest}/{settings.Patience} | Restante: {remaining:hh\\:mm\\:ss}   ");

                            if ((epoch + 1) % 10 == 0 || epoch + 1 == totalEpochs)
                            {
                                Console.WriteLine();
                            }

                            if (valLoss < 1.0 && callbackCount % 10 == 0)
                            {
                                Console.WriteLine();
                                Console.WriteLine($"  HITO: pérdida de validación por debajo de 1.0 en la época {epoch + 1}");
                                Console.WriteLine();
                            }

                            if (valLoss < 0.5 && callbackCount % 10 == 0)
                            {
                                Console.WriteLine();
                                Console.WriteLine($"  EXCELENTE: pérdida de validación por debajo de 0.5 en la época {epoch + 1}");
                                Console.WriteLine();
                            }
                        }
                    );

                    Console.WriteLine();
                    Console.WriteLine($"¡Entrenamiento completado! (Épocas en esta corrida: {callbackCount})");
                    Console.WriteLine($"Mejor pérdida de validación alcanzada: {bestLoss:F6}");

                    if (bestLoss > 2.0)
                    {
                        Console.WriteLine();
                        Console.WriteLine("ADVERTENCIA: la pérdida sigue alta (>2.0)");
                        Console.WriteLine("Es posible que el modelo no haya aprendido bien los patrones.");
                        Console.WriteLine("Considera:");
                        Console.WriteLine("  1. Entrenar más épocas");
                        Console.WriteLine("  2. Agregar ejemplos de entrenamiento más variados");
                        Console.WriteLine("  3. Verificar que el formato de los datos sea correcto");
                    }
                    else if (bestLoss > 1.0)
                    {
                        Console.WriteLine();
                        Console.WriteLine("AVISO: la pérdida es moderada (1.0-2.0)");
                        Console.WriteLine("El modelo aprendió algunos patrones, pero podría mejorar con más entrenamiento.");
                    }
                    else if (bestLoss > 0.5)
                    {
                        Console.WriteLine();
                        Console.WriteLine("BIEN: la pérdida es baja (0.5-1.0)");
                        Console.WriteLine("El modelo debería generar respuestas razonables.");
                    }
                    else
                    {
                        Console.WriteLine();
                        Console.WriteLine("EXCELENTE: la pérdida es muy baja (<0.5)");
                        Console.WriteLine("¡El modelo aprendió bien los patrones!");
                    }

                    Console.WriteLine();

                    monitor.PrintFinalStats();
                }

                Console.WriteLine($"Guardando el modelo entrenado en: {modelPath}");
                chatBot.SaveCompleteModel(modelPath);
                TransformerChatBotService.DeleteTrainingProgress(modelPath);
                Console.WriteLine("¡Modelo guardado correctamente!");
                Console.WriteLine();

                StartChat(chatBot).GetAwaiter().GetResult();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        static void RunChatOnly(string[] args)
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("       Neuraval — Chat de prueba");
            Console.WriteLine("===========================================");
            Console.WriteLine();

            var settings = TrainingSettings.Load("training-settings.json", args);
            string modelPath = settings.ModelPath;

            if (!Directory.Exists(modelPath))
            {
                Console.WriteLine($"No se encontró ningún modelo guardado en '{modelPath}'.");
                Console.WriteLine("Entrená primero con: dotnet run --project Neuraval.CLI");
                return;
            }

            int numThreads = settings.NumThreads > 0 ? settings.NumThreads : Environment.ProcessorCount;

            var chatBot = new TransformerChatBotService(
                embeddingDim: settings.EmbeddingDim,
                numLayers: settings.NumLayers,
                numHeads: settings.NumHeads,
                feedforwardDim: settings.FeedforwardDim,
                maxSequenceLength: settings.MaxSequenceLength,
                dropout: settings.Dropout,
                numThreads: numThreads
            );

            Console.WriteLine("Cargando modelo...");
            Console.WriteLine();

            bool modelLoaded = chatBot.LoadCompleteModel(modelPath);

            if (!modelLoaded)
            {
                Console.WriteLine("No se pudo cargar el modelo. Revisa que la carpeta tenga model.navm y tokenizer.json.");
                return;
            }

            if (!chatBot.IsTrained())
            {
                Console.WriteLine("AVISO: este checkpoint quedó a mitad de entrenamiento (nunca completó una corrida entera).");
                Console.WriteLine("Las respuestas pueden estar poco pulidas todavía.");
                Console.WriteLine();
            }

            Console.WriteLine("¡Modelo cargado! (modo solo chat, no se va a entrenar nada)");
            Console.WriteLine();

            StartChat(chatBot).GetAwaiter().GetResult();
        }

        static void RunConvert(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("Uso: dotnet run --project Neuraval.CLI -- --convert <entrada.json> <salida.txt>");
                return;
            }

            string jsonPath = args[1];
            string txtPath = args[2];

            try
            {
                DatasetLoader.ConvertIndexedJsonToPlainText(jsonPath, txtPath);
                Console.WriteLine($"Convertido: {jsonPath} -> {txtPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al convertir {jsonPath}: {ex.Message}");
            }
        }

        static void RunInt8Benchmark(string[] args)
        {
            // Por defecto usa las dimensiones del preset "medium" de 5.2,
            // que es un tamaño representativo de proyección de atención
            // (embeddingDim x embeddingDim). Se pueden pasar dimensiones
            // propias: --int8-benchmark <embeddingDim> <seqLen> <iteraciones>
            int embeddingDim = args.Length > 1 && int.TryParse(args[1], out var d) ? d : 512;
            int seqLen = args.Length > 2 && int.TryParse(args[2], out var s) ? s : 512;
            int iterations = args.Length > 3 && int.TryParse(args[3], out var it) ? it : 20;

            Console.WriteLine("=== Benchmark matmul CPU: FP32 (SIMD existente) vs INT8 (Fase 5.4.2) ===");
            Console.WriteLine($"embeddingDim={embeddingDim}, seqLen={seqLen}, iteraciones={iterations}");
            Console.WriteLine();

            var random = new Random(42);
            var weights = new float[embeddingDim, embeddingDim];
            for (int i = 0; i < embeddingDim; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    weights[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }

            var inputArr = new float[seqLen, embeddingDim];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < embeddingDim; j++)
                {
                    inputArr[i, j] = (float)(random.NextDouble() * 2.0 - 1.0);
                }
            }

            var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(inputArr, DeviceType.Cpu);
            var fp32Cache = new CudaWeightCache(embeddingDim, embeddingDim);
            var int8Cache = new Int8WeightCache(embeddingDim, embeddingDim);

            // Un llamado de precalentamiento de cada uno (JIT warm-up), fuera de la medición.
            var fp32Warmup = TensorOps.MatMulCachedB(inputTensor, weights, fp32Cache);
            var int8Warmup = Int8MatMul.MatMulCachedB(inputTensor, weights, int8Cache);

            var (maxError, meanError) = Int8Quantizer.ComputeQuantizationError(fp32Warmup.Buffer, int8Warmup.Buffer);

            var swFp32 = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                TensorOps.MatMulCachedB(inputTensor, weights, fp32Cache);
            }
            swFp32.Stop();

            var swInt8 = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                Int8MatMul.MatMulCachedB(inputTensor, weights, int8Cache);
            }
            swInt8.Stop();

            double fp32MsPerCall = swFp32.Elapsed.TotalMilliseconds / iterations;
            double int8MsPerCall = swInt8.Elapsed.TotalMilliseconds / iterations;
            double speedup = fp32MsPerCall / int8MsPerCall;

            long fp32Bytes = (long)embeddingDim * embeddingDim * sizeof(float);
            long int8Bytes = (long)embeddingDim * embeddingDim * sizeof(sbyte) + embeddingDim * sizeof(float);

            Console.WriteLine($"FP32 (backend SIMD existente): {fp32MsPerCall:F3} ms/llamada");
            Console.WriteLine($"INT8 (kernel escalar 5.4.2):   {int8MsPerCall:F3} ms/llamada");
            Console.WriteLine($"Relación de velocidad INT8/FP32: {speedup:F2}x ({(speedup >= 1.0 ? "más rápido" : "más lento")})");
            Console.WriteLine();
            Console.WriteLine($"Peso en memoria por matriz — FP32: {fp32Bytes / 1024.0:F1} KB, INT8: {int8Bytes / 1024.0:F1} KB " +
                $"({(1.0 - (double)int8Bytes / fp32Bytes) * 100.0:F1}% menos)");
            Console.WriteLine($"Error de cuantización en la salida — máximo: {maxError:F6}, promedio: {meanError:F6}");
            Console.WriteLine();
            Console.WriteLine("Si INT8 no resultó más rápido en esta máquina, dejalo deshabilitado " +
                "(es el valor por defecto): el ahorro garantizado de esta fase es memoria, no velocidad.");
        }

        static void RunQuantize(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("Uso: dotnet run --project Neuraval.CLI -- --quantize <modelo.navm> <modelo-int8.navm>");
                return;
            }

            string inputPath = args[1];
            string outputPath = args[2];

            if (!File.Exists(inputPath))
            {
                Console.WriteLine($"No se encontró el archivo: {inputPath}");
                return;
            }

            try
            {
                Console.WriteLine($"Cargando modelo: {inputPath}");
                var (modelState, header) = ModelBinarySerializer.Load(inputPath);

                if (header.Quantized)
                {
                    Console.WriteLine("El modelo de entrada ya está cuantizado en INT8; no hay nada que hacer.");
                    return;
                }

                if (!header.IsTrained)
                {
                    Console.WriteLine("Advertencia: el modelo no figura como entrenado (IsTrained=false). Se cuantiza igual.");
                }

                // Diagnóstico: error de cuantización sobre la matriz más
                // grande (embeddings, compartida con la proyección de
                // salida por weight tying), que es la que más pesa y la más
                // representativa del impacto en precisión.
                var embeddings = modelState.EmbeddingState.Embeddings;
                var quantizedEmbeddings = Int8Quantizer.QuantizeRowSymmetric(
                    embeddings, modelState.EmbeddingState.VocabSize, modelState.EmbeddingState.EmbeddingDim);
                var dequantizedEmbeddings = Int8Quantizer.Dequantize(quantizedEmbeddings);
                var (maxError, meanError) = Int8Quantizer.ComputeQuantizationError(embeddings, dequantizedEmbeddings);

                long originalSize = new FileInfo(inputPath).Length;

                Console.WriteLine($"Cuantizando y guardando: {outputPath}");
                ModelBinarySerializer.SaveQuantized(outputPath, modelState, header);

                long quantizedSize = new FileInfo(outputPath).Length;
                double reductionPercent = originalSize > 0
                    ? (1.0 - (double)quantizedSize / originalSize) * 100.0
                    : 0.0;

                Console.WriteLine();
                Console.WriteLine("=== Cuantización INT8 completada ===");
                Console.WriteLine($"Tamaño original:    {originalSize / 1024.0 / 1024.0:F2} MB");
                Console.WriteLine($"Tamaño cuantizado:  {quantizedSize / 1024.0 / 1024.0:F2} MB");
                Console.WriteLine($"Reducción:          {reductionPercent:F1}%");
                Console.WriteLine($"Error cuantización (embeddings) — máximo: {maxError:F6}, promedio: {meanError:F6}");
                Console.WriteLine();
                Console.WriteLine("El archivo cuantizado es solo para inferencia (no conserva estado de optimizadores).");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al cuantizar {inputPath}: {ex.Message}");
            }
        }

        /// <summary>
        /// Habilita LoRA sobre el modelo del <paramref name="chatBot"/> si
        /// <see cref="TrainingSettings.LoraRank"/> &gt; 0. Es un no-op si el
        /// modelo ya trae LoRA habilitado (por ejemplo, porque se cargó desde
        /// un checkpoint que ya lo tenía).
        /// </summary>
        static void ApplyLoraSettingsIfRequested(TransformerChatBotService chatBot, TrainingSettings settings)
        {
            if (settings.LoraRank <= 0)
            {
                return;
            }

            var model = chatBot.GetModel();
            if (model == null)
            {
                Console.WriteLine("Advertencia: se pidió LoRA pero el modelo todavía no está inicializado.");
                return;
            }

            if (model.HasLora)
            {
                Console.WriteLine("El modelo ya tiene LoRA habilitado (viene del checkpoint); se continúa con los adaptadores existentes.");
                Console.WriteLine();
                return;
            }

            model.EnableLora(settings.LoraRank, (float)settings.LoraAlpha, freezeBase: true);

            Console.WriteLine($"LoRA habilitado: rank={settings.LoraRank}, alpha={settings.LoraAlpha}.");
            Console.WriteLine("La base del modelo quedó congelada; solo se van a entrenar los adaptadores A/B de atención.");
            Console.WriteLine();
        }

        static void RunExportLora(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("Uso: dotnet run --project Neuraval.CLI -- --export-lora <modelo.navm> <adaptador.navlora>");
                return;
            }

            string modelPath = args[1];
            string loraOutputPath = args[2];

            if (!File.Exists(modelPath))
            {
                Console.WriteLine($"No se encontró el archivo: {modelPath}");
                return;
            }

            try
            {
                Console.WriteLine($"Cargando modelo: {modelPath}");
                var (modelState, _) = ModelBinarySerializer.Load(modelPath);
                var model = TransformerModel.LoadState(modelState);

                var loraState = model.SaveLoraState();
                if (loraState == null)
                {
                    Console.WriteLine("El modelo no tiene adaptadores LoRA habilitados; no hay nada que exportar.");
                    Console.WriteLine("(¿Se entrenó con --lora-rank?)");
                    return;
                }

                Console.WriteLine($"Exportando adaptador LoRA: {loraOutputPath}");
                LoraBinarySerializer.Save(loraOutputPath, loraState);

                long loraSize = new FileInfo(loraOutputPath).Length;
                long modelSize = new FileInfo(modelPath).Length;

                Console.WriteLine();
                Console.WriteLine("=== Exportación de adaptador LoRA completada ===");
                Console.WriteLine($"Modelo base:        {modelSize / 1024.0 / 1024.0:F2} MB");
                Console.WriteLine($"Adaptador LoRA:     {loraSize / 1024.0 / 1024.0:F2} MB");
                Console.WriteLine($"Rank: {loraState.BlockStates[0].Query.Rank}, Alpha: {loraState.BlockStates[0].Query.Alpha}, Capas: {loraState.NumLayers}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al exportar LoRA desde {modelPath}: {ex.Message}");
            }
        }

        static void RunImportLora(string[] args)
        {
            if (args.Length < 4)
            {
                Console.WriteLine("Uso: dotnet run --project Neuraval.CLI -- --import-lora <modelo-base.navm> <adaptador.navlora> <modelo-salida.navm>");
                return;
            }

            string basePath = args[1];
            string loraPath = args[2];
            string outputPath = args[3];

            if (!File.Exists(basePath))
            {
                Console.WriteLine($"No se encontró el archivo: {basePath}");
                return;
            }

            if (!File.Exists(loraPath))
            {
                Console.WriteLine($"No se encontró el archivo: {loraPath}");
                return;
            }

            try
            {
                Console.WriteLine($"Cargando modelo base: {basePath}");
                var (modelState, header) = ModelBinarySerializer.Load(basePath);

                if (header.Quantized)
                {
                    Console.WriteLine("El modelo base está cuantizado en INT8; importar LoRA requiere el .navm sin cuantizar (con estado de optimizadores).");
                    return;
                }

                var model = TransformerModel.LoadState(modelState);

                Console.WriteLine($"Cargando adaptador LoRA: {loraPath}");
                var (loraState, loraHeader) = LoraBinarySerializer.Load(loraPath);

                if (loraHeader.EmbeddingDim != header.EmbeddingDim || loraHeader.NumLayers != header.NumLayers)
                {
                    Console.WriteLine(
                        $"El adaptador ({loraHeader.NumLayers} capas, embeddingDim={loraHeader.EmbeddingDim}) " +
                        $"no es compatible con el modelo base ({header.NumLayers} capas, embeddingDim={header.EmbeddingDim}).");
                    return;
                }

                model.LoadLoraState(loraState);

                var mergedState = model.SaveState();
                Console.WriteLine($"Guardando modelo con LoRA importado: {outputPath}");
                ModelBinarySerializer.Save(outputPath, mergedState, header);

                Console.WriteLine();
                Console.WriteLine("=== Importación de adaptador LoRA completada ===");
                Console.WriteLine($"Rank: {loraHeader.Rank}, Alpha: {loraHeader.Alpha}, Base congelada: {loraHeader.FreezeNonLoraWeights}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al importar LoRA hacia {basePath}: {ex.Message}");
            }
        }

        static void RunConvertAll(string[] args)
        {
            string folder = args.Length > 1 ? args[1] : "Data";
            bool deleteOriginal = args.Contains("--delete-original");

            if (!Directory.Exists(folder))
            {
                Console.WriteLine($"Carpeta no encontrada: {folder}");
                return;
            }

            var jsonFiles = Directory.GetFiles(folder, "*.json", SearchOption.AllDirectories);

            if (jsonFiles.Length == 0)
            {
                Console.WriteLine($"No se encontraron archivos .json en {folder}.");
                return;
            }

            foreach (var jsonFile in jsonFiles)
            {
                var txtFile = Path.ChangeExtension(jsonFile, ".txt");

                try
                {
                    DatasetLoader.ConvertIndexedJsonToPlainText(jsonFile, txtFile);
                    Console.WriteLine($"Convertido: {jsonFile} -> {txtFile}");

                    if (deleteOriginal)
                    {
                        File.Delete(jsonFile);
                        Console.WriteLine($"  Eliminado original: {jsonFile}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error al convertir {jsonFile}: {ex.Message}");
                }
            }
        }

        static void RunTrainingBenchmark()
        {
            int embeddingDim = 64;
            int numLayers = 2;
            int numHeads = 4;
            int feedforwardDim = 256;
            int maxSequenceLength = 32;
            int epochs = 5;
            int batchSize = 8;

            var tokenizer = new Tokenizer();
            var texts = new List<string>();
            var pairs = new List<(string prompt, string response)>();
            var rng = new Random(42);
            string[] words = { "hola", "como", "estas", "bien", "gracias", "que", "tal", "adios", "nos", "vemos", "cuidate", "luego", "todo", "aqui", "buenos", "dias" };

            for (int i = 0; i < 200; i++)
            {
                int promptLen = rng.Next(2, 6);
                int responseLen = rng.Next(2, 8);
                var prompt = string.Join(" ", Enumerable.Range(0, promptLen).Select(_ => words[rng.Next(words.Length)]));
                var response = string.Join(" ", Enumerable.Range(0, responseLen).Select(_ => words[rng.Next(words.Length)]));
                pairs.Add((prompt, response));
                texts.Add(prompt);
                texts.Add(response);
            }

            tokenizer.BuildVocabulary(texts, minFrequency: 1, maxVocabSize: 500);

            var datasetLoader = new DatasetLoader(tokenizer, maxSequenceLength: maxSequenceLength);
            var examples = pairs
                .Select(p => datasetLoader.BuildCausalExample(p.prompt, p.response))
                .Where(e => e.ResponseStartIndex >= 0)
                .ToList();

            var model = new TransformerModel(
                vocabSize: tokenizer.VocabSize,
                embeddingDim: embeddingDim,
                numLayers: numLayers,
                numHeads: numHeads,
                feedforwardDim: feedforwardDim,
                maxSequenceLength: maxSequenceLength,
                dropout: 0.0f,
                seed: 1);

            var trainer = new SupervisedTrainer(model, learningRate: 0.001f, padToken: tokenizer.PadToken);

            Console.WriteLine($"Benchmark: {examples.Count} ejemplos, batchSize={batchSize}, epochs={epochs}, hilos={Matematicas.GetNumThreads()}");

            GC.Collect();
            int gen0Before = GC.CollectionCount(0);
            int gen1Before = GC.CollectionCount(1);
            int gen2Before = GC.CollectionCount(2);

            var stopwatch = Stopwatch.StartNew();

            trainer.TrainCausalWithValidation(
                trainingExamples: examples,
                validationExamples: examples,
                epochs: epochs,
                batchSize: batchSize,
                patience: epochs);

            stopwatch.Stop();

            int gen0After = GC.CollectionCount(0);
            int gen1After = GC.CollectionCount(1);
            int gen2After = GC.CollectionCount(2);

            Console.WriteLine($"Tiempo total: {stopwatch.ElapsedMilliseconds} ms");
            Console.WriteLine($"Tiempo por epoca: {stopwatch.ElapsedMilliseconds / (double)epochs:F1} ms");
            Console.WriteLine($"GC gen0: {gen0After - gen0Before}, gen1: {gen1After - gen1Before}, gen2: {gen2After - gen2Before}");
        }

        private sealed class GpuBenchResult
        {
            public string Op { get; init; } = "";
            public string Shape { get; init; } = "";
            public long ElementCount { get; init; }
            public double CpuMs { get; init; }
            public double? GpuMs { get; init; }
        }

        /// <summary>
        /// Checkpoint 6.3.8: microbenchmark CPU vs GPU para cada una de las operaciones que
        /// tienen camino GPU-con-fallback (matmul, matmul transpose A/B, softmax por fila,
        /// layernorm por fila, paso de Adam), a varios tamaños representativos del modelo
        /// (embeddingDim, hiddenDim, seqLen, vocabSize). No entrena nada: mide cada operación
        /// aislada, con GC forzado y una iteración de "warmup" antes de cronometrar, para dar
        /// un umbral de tamaño real a partir del cual conviene activar UseGpu.
        /// </summary>
        static void RunGpuBenchmark()
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("   Benchmark GPU vs CPU (checkpoint 6.3.8)");
            Console.WriteLine("===========================================");
            Console.WriteLine();

            if (!Matematicas.IsGpuAvailable())
            {
                Console.WriteLine("No se detectó ninguna GPU CUDA (o falta compilar/copiar navcuda en esta máquina).");
                Console.WriteLine("Compila Neuraval.Cuda.Native (build.sh / build.bat) y volvé a correr --gpu-benchmark.");
                return;
            }

            if (!Matematicas.TryEnableGpu())
            {
                Console.WriteLine("Se detectó una GPU pero no se pudo inicializar CUDA. Revisá el driver y volvé a intentar.");
                return;
            }

            Console.WriteLine("GPU CUDA detectada y activada. Corriendo microbenchmarks...");
            Console.WriteLine("(esto puede tardar varios minutos: cada combinación de tamaño corre 1 warmup + 5 repeticiones en CPU y en GPU)");
            Console.WriteLine();

            var results = new List<GpuBenchResult>();
            var rng = new Random(1234);
            const int iterations = 5;

            double Time(Action action)
            {
                action(); // warmup, no cuenta (primer kernel launch / primera transferencia son más lentos)

                GC.Collect();
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                {
                    action();
                }
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds / iterations;
            }

            void Bench(string op, string shape, long elementCount, Action cpuAction, Action gpuAction)
            {
                double cpuMs = Time(cpuAction);
                double? gpuMs;

                try
                {
                    gpuMs = Time(gpuAction);
                }
                catch (CudaException ex)
                {
                    Console.WriteLine($"  [{op} {shape}] GPU falló ({ex.Message}), se registra solo el tiempo CPU.");
                    gpuMs = null;
                }

                results.Add(new GpuBenchResult { Op = op, Shape = shape, ElementCount = elementCount, CpuMs = cpuMs, GpuMs = gpuMs });
            }

            // Tamaños representativos: embeddingDim / hiddenDim como en modelos chicos-medianos-grandes,
            // seqLen típico de entrenamiento por lotes, y vocabSize para la tabla de embeddings.
            int[] embeddingDims = { 64, 128, 256, 512, 1024 };
            int[] seqLens = { 32, 128, 512 };
            int[] vocabSizes = { 500, 5000, 30000 };

            // --- MatMul (A·B): proyecciones Q/K/V/salida (seqLen x embeddingDim) · (embeddingDim x embeddingDim) ---
            foreach (int dim in embeddingDims)
            {
                foreach (int seqLen in seqLens)
                {
                    var a = RandomMatrix(seqLen, dim, rng);
                    var b = RandomMatrix(dim, dim, rng);

                    Bench("MatMul A·B", $"({seqLen}x{dim})·({dim}x{dim})", (long)seqLen * dim * dim,
                        () => Matematicas.ParallelMatrixMultiply(a, b),
                        () => CudaMath.MatrixMultiply(a, b));
                }
            }

            // --- MatMul transpuesta B (Q·Kᵀ escalado): (seqLen x dim) · (seqLen x dim)ᵀ ---
            foreach (int dim in embeddingDims)
            {
                foreach (int seqLen in seqLens)
                {
                    var a = RandomMatrix(seqLen, dim, rng);
                    var b = RandomMatrix(seqLen, dim, rng);
                    float scale = 1.0f / MathF.Sqrt(dim);

                    Bench("MatMul A·Bᵀ", $"({seqLen}x{dim})·({seqLen}x{dim})ᵀ", (long)seqLen * seqLen * dim,
                        () => Matematicas.ParallelMatrixMultiplyTransposeB(a, b, scale),
                        () => CudaMath.MatrixMultiplyTransposeB(a, b, scale));
                }
            }

            // --- MatMul transpuesta A (gradiente de pesos, Aᵀ·B): (seqLen x dim)ᵀ · (seqLen x dim) ---
            foreach (int dim in embeddingDims)
            {
                foreach (int seqLen in seqLens)
                {
                    var a = RandomMatrix(seqLen, dim, rng);
                    var b = RandomMatrix(seqLen, dim, rng);

                    Bench("MatMul Aᵀ·B", $"({seqLen}x{dim})ᵀ·({seqLen}x{dim})", (long)seqLen * dim * dim,
                        () => Matematicas.ParallelMatrixMultiplyTransposeA(a, b),
                        () => CudaMath.MatrixMultiplyTransposeA(a, b));
                }
            }

            // --- Softmax por fila: una fila por posición de secuencia (o por cabeza), cols = seqLen ---
            foreach (int seqLen in seqLens)
            {
                var input = RandomMatrix(seqLen, seqLen, rng);

                Bench("SoftmaxRows", $"({seqLen}x{seqLen})", (long)seqLen * seqLen,
                    () => Matematicas.ParallelSoftmaxRows(input),
                    () => CudaMath.SoftmaxRows(input));
            }

            // --- LayerNorm por fila: cols = embeddingDim ---
            foreach (int dim in embeddingDims)
            {
                foreach (int seqLen in seqLens)
                {
                    var input = RandomMatrix(seqLen, dim, rng);
                    var gamma = RandomVector(dim, rng);
                    var beta = RandomVector(dim, rng);

                    Bench("LayerNormRows", $"({seqLen}x{dim})", (long)seqLen * dim,
                        () => Matematicas.ParallelLayerNormRows(input, gamma, beta, 1e-5f, out _, out _),
                        () => CudaMath.LayerNormRows(input, gamma, beta, 1e-5f, out _, out _));
                }
            }

            // --- Paso de Adam: elemento a elemento sobre pesos de feedforward y tabla de embeddings ---
            foreach (int dim in embeddingDims)
            {
                int hiddenDim = dim * 4;
                var parametersCpu = RandomMatrix(dim, hiddenDim, rng);
                var gradientsCpu = RandomMatrix(dim, hiddenDim, rng);
                var mCpu = new float[dim, hiddenDim];
                var vCpu = new float[dim, hiddenDim];

                var parametersGpu = (float[,])parametersCpu.Clone();
                var gradientsGpu = (float[,])gradientsCpu.Clone();
                var mGpu = new float[dim, hiddenDim];
                var vGpu = new float[dim, hiddenDim];

                Bench("AdamUpdate (feedforward)", $"({dim}x{hiddenDim})", (long)dim * hiddenDim,
                    () => Matematicas.ParallelAdamUpdate(parametersCpu, gradientsCpu, mCpu, vCpu, 0.9f, 0.999f, 1e-8f, 0.001f, 0.9f, 0.999f),
                    () => CudaMath.AdamUpdate(parametersGpu, gradientsGpu, mGpu, vGpu, 0.9f, 0.999f, 1e-8f, 0.001f, 0.9f, 0.999f));
            }

            foreach (int vocabSize in vocabSizes)
            {
                int dim = 256;
                var parametersCpu = RandomMatrix(vocabSize, dim, rng);
                var gradientsCpu = RandomMatrix(vocabSize, dim, rng);
                var mCpu = new float[vocabSize, dim];
                var vCpu = new float[vocabSize, dim];

                var parametersGpu = (float[,])parametersCpu.Clone();
                var gradientsGpu = (float[,])gradientsCpu.Clone();
                var mGpu = new float[vocabSize, dim];
                var vGpu = new float[vocabSize, dim];

                Bench("AdamUpdate (embeddings)", $"({vocabSize}x{dim})", (long)vocabSize * dim,
                    () => Matematicas.ParallelAdamUpdate(parametersCpu, gradientsCpu, mCpu, vCpu, 0.9f, 0.999f, 1e-8f, 0.001f, 0.9f, 0.999f),
                    () => CudaMath.AdamUpdate(parametersGpu, gradientsGpu, mGpu, vGpu, 0.9f, 0.999f, 1e-8f, 0.001f, 0.9f, 0.999f));
            }

            PrintGpuBenchmarkResults(results);
        }

        private sealed class SuiteBenchRow
        {
            public string Operation { get; init; } = "";
            public string Shape { get; init; } = "";
            public double SequentialMs { get; init; }
            public double ParallelMs { get; init; }
            public double? CudaMs { get; init; }
        }

        static void RunBenchmarkSuite()
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("   Suite de benchmarks (Fase 1, item 18)");
            Console.WriteLine("===========================================");
            Console.WriteLine();

            bool gpuAvailable = Matematicas.IsGpuAvailable() && Matematicas.TryEnableGpu();
            Console.WriteLine(gpuAvailable ? "GPU CUDA detectada y activada." : "GPU CUDA no disponible: sólo se medirá CPU.");
            Console.WriteLine();

            var rng = new Random(2026);
            var rows = new List<SuiteBenchRow>();

            double Time(Action action, int iterations = 5)
            {
                action();
                GC.Collect();
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                {
                    action();
                }
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds / iterations;
            }

            double? TimeCuda(Action action, int iterations = 5)
            {
                if (!gpuAvailable) return null;
                try
                {
                    return Time(action, iterations);
                }
                catch (CudaException)
                {
                    return null;
                }
            }

            int[] matMulSizes = { 128, 512, 1024 };
            foreach (int size in matMulSizes)
            {
                var a = RandomMatrix(size, size, rng);
                var b = RandomMatrix(size, size, rng);
                int iterations = size >= 1024 ? 2 : 5;

                double seqMs = Time(() => Matematicas.SequentialMatrixMultiply(a, b), iterations);
                double parMs = Time(() => Matematicas.ParallelMatrixMultiply(a, b), iterations);
                double? cudaMs = TimeCuda(() => CudaMath.MatrixMultiply(a, b), iterations);

                rows.Add(new SuiteBenchRow { Operation = "MatMul", Shape = $"{size}x{size}", SequentialMs = seqMs, ParallelMs = parMs, CudaMs = cudaMs });
            }

            int[] softmaxSizes = { 128, 512, 1024 };
            foreach (int size in softmaxSizes)
            {
                var input = RandomMatrix(size, size, rng);

                double seqMs = Time(() => Matematicas.SequentialSoftmax2D(input));
                double parMs = Time(() => Matematicas.ParallelSoftmaxRows(input));
                double? cudaMs = TimeCuda(() => CudaMath.SoftmaxRows(input));

                rows.Add(new SuiteBenchRow { Operation = "Softmax", Shape = $"{size}x{size}", SequentialMs = seqMs, ParallelMs = parMs, CudaMs = cudaMs });
            }

            int[] layerNormDims = { 128, 512, 1024 };
            foreach (int dim in layerNormDims)
            {
                int seqLen = 128;
                var input = RandomMatrix(seqLen, dim, rng);
                var gamma = RandomVector(dim, rng);
                var beta = RandomVector(dim, rng);

                double parMs = Time(() => Matematicas.ParallelLayerNormRows(input, gamma, beta, 1e-5f, out _, out _));
                double? cudaMs = TimeCuda(() => CudaMath.LayerNormRows(input, gamma, beta, 1e-5f, out _, out _));

                rows.Add(new SuiteBenchRow { Operation = "LayerNorm", Shape = $"{seqLen}x{dim}", SequentialMs = double.NaN, ParallelMs = parMs, CudaMs = cudaMs });
            }

            foreach (var (label, embeddingDim, numLayers, numHeads, feedforwardDim, seqLen) in new[]
            {
                ("Nano", 128, 2, 4, 512, 32),
                ("Small", 256, 6, 8, 1024, 128),
            })
            {
                var model = new TransformerModel(
                    vocabSize: 1000,
                    embeddingDim: embeddingDim,
                    numLayers: numLayers,
                    numHeads: numHeads,
                    feedforwardDim: feedforwardDim,
                    maxSequenceLength: seqLen,
                    dropout: 0.0f,
                    seed: 1);

                var tokens = Enumerable.Range(0, seqLen).Select(i => i % 1000).ToArray();

                double forwardMs = Time(() => model.Forward(tokens, training: false), iterations: 3);
                rows.Add(new SuiteBenchRow { Operation = $"Transformer Forward ({label})", Shape = $"seq={seqLen}, dim={embeddingDim}, layers={numLayers}", SequentialMs = double.NaN, ParallelMs = forwardMs, CudaMs = null });

                double trainStepMs = Time(() =>
                {
                    model.ZeroGradients();
                    model.CalculateCausalLoss(tokens, lossStartIndex: 0);
                    model.AverageGradients(1);
                    model.ClipGradients(1.0f);
                    model.UpdateWeights(0.001f);
                }, iterations: 3);
                rows.Add(new SuiteBenchRow { Operation = $"Transformer Training step ({label})", Shape = $"seq={seqLen}, dim={embeddingDim}, layers={numLayers}", SequentialMs = double.NaN, ParallelMs = trainStepMs, CudaMs = null });
            }

            PrintBenchmarkSuiteResults(rows);
        }

        static void PrintBenchmarkSuiteResults(List<SuiteBenchRow> rows)
        {
            Console.WriteLine();
            Console.WriteLine($"{"Operación",-30} {"Forma",-30} {"Secuencial (ms)",16} {"Paralelo (ms)",14} {"CUDA (ms)",10}");
            Console.WriteLine(new string('-', 106));

            foreach (var r in rows)
            {
                string seqStr = double.IsNaN(r.SequentialMs) ? "n/a" : r.SequentialMs.ToString("F3");
                string cudaStr = r.CudaMs.HasValue ? r.CudaMs.Value.ToString("F3") : "n/a";

                Console.WriteLine($"{r.Operation,-30} {r.Shape,-30} {seqStr,16} {r.ParallelMs,14:F3} {cudaStr,10}");
            }

            Console.WriteLine();
        }

        static float[,] RandomMatrix(int rows, int cols, Random rng)
        {
            var m = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m[i, j] = (float)(rng.NextDouble() * 2.0 - 1.0);
                }
            }
            return m;
        }

        static float[] RandomVector(int n, Random rng)
        {
            var v = new float[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            return v;
        }

        static void RunMemoryProfile()
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("   Memory profiling (Fase 1, item 4)");
            Console.WriteLine("===========================================");
            Console.WriteLine();

            int vocabSize = 1000;

            foreach (var (label, embeddingDim, numLayers, numHeads, feedforwardDim, seqLen) in new[]
            {
                ("Nano", 128, 2, 4, 512, 32),
                ("Small", 256, 6, 8, 1024, 128),
            })
            {
                var report = MemoryProfiler.Analyze(
                    vocabSize: vocabSize,
                    embeddingDim: embeddingDim,
                    numLayers: numLayers,
                    numHeads: numHeads,
                    feedforwardDim: feedforwardDim,
                    batchSize: 1,
                    sequenceLength: seqLen);

                PrintMemoryReport(label, report);
                RunEmpiricalMemoryMeasurement(label, vocabSize, embeddingDim, numLayers, numHeads, feedforwardDim, seqLen);
            }
        }

        static void PrintMemoryReport(string label, ModelMemoryReport report)
        {
            Console.WriteLine($"--- {label} (estimación teórica) ---");
            Console.WriteLine($"{"Componente",-26} {"Parámetros",14} {"Pesos",10} {"Gradientes",12} {"Optimizer",10} {"Total",10}");
            Console.WriteLine(new string('-', 88));

            foreach (var component in report.Parameters)
            {
                Console.WriteLine($"{component.Name,-26} {component.ParameterCount,14:N0} {MemoryProfiler.FormatBytes(component.WeightsBytes),10} {MemoryProfiler.FormatBytes(component.GradientBytes),12} {MemoryProfiler.FormatBytes(component.OptimizerStateBytes),10} {MemoryProfiler.FormatBytes(component.TotalBytes),10}");
            }

            Console.WriteLine(new string('-', 88));
            Console.WriteLine($"{"Total parámetros",-26} {report.TotalParameterCount,14:N0} {MemoryProfiler.FormatBytes(report.TotalWeightsBytes),10} {MemoryProfiler.FormatBytes(report.TotalGradientBytes),12} {MemoryProfiler.FormatBytes(report.TotalOptimizerStateBytes),10} {MemoryProfiler.FormatBytes(report.TotalParameterBytes),10}");
            Console.WriteLine();

            Console.WriteLine("Activaciones estimadas (batch=1):");
            foreach (var activation in report.Activations)
            {
                Console.WriteLine($"  {activation.Name,-26} {activation.ElementCount,14:N0} {MemoryProfiler.FormatBytes(activation.Bytes),10}");
            }
            Console.WriteLine($"  {"Total activaciones",-26} {"",14} {"",10} {"",12} {"",10} {MemoryProfiler.FormatBytes(report.TotalActivationBytes),10}");
            Console.WriteLine();

            double adamRatio = report.TotalWeightsBytes > 0
                ? (double)(report.TotalWeightsBytes + report.TotalOptimizerStateBytes) / report.TotalWeightsBytes
                : 0;
            double fullRatio = report.TotalWeightsBytes > 0
                ? (double)report.TotalParameterBytes / report.TotalWeightsBytes
                : 0;

            Console.WriteLine($"Regla teórica de Adam (pesos + m + v): {adamRatio:F1}x el tamaño de los pesos");
            Console.WriteLine($"Consumo real en Neuraval (pesos + gradientes + accumulated + m + v): {fullRatio:F1}x el tamaño de los pesos");
            Console.WriteLine($"Total estimado (parámetros + activaciones): {MemoryProfiler.FormatBytes(report.GrandTotalBytes)}");
            Console.WriteLine();
        }

        static void RunEmpiricalMemoryMeasurement(
            string label,
            int vocabSize,
            int embeddingDim,
            int numLayers,
            int numHeads,
            int feedforwardDim,
            int seqLen)
        {
            Console.WriteLine($"--- {label} (medido en runtime con GC.GetTotalMemory) ---");

            long beforeConstruct = ForceCollectAndMeasure();

            var model = new TransformerModel(
                vocabSize: vocabSize,
                embeddingDim: embeddingDim,
                numLayers: numLayers,
                numHeads: numHeads,
                feedforwardDim: feedforwardDim,
                maxSequenceLength: seqLen,
                dropout: 0.0f,
                seed: 1);

            long afterConstruct = ForceCollectAndMeasure();

            var tokens = Enumerable.Range(0, seqLen).Select(i => i % vocabSize).ToArray();

            model.Forward(tokens, training: true);
            long afterForward = ForceCollectAndMeasure();

            model.ZeroGradients();
            model.CalculateCausalLoss(tokens, lossStartIndex: 0);
            model.AverageGradients(1);
            model.ClipGradients(1.0f);
            long afterBackward = ForceCollectAndMeasure();

            model.UpdateWeights(0.001f);
            long afterUpdate = ForceCollectAndMeasure();

            Console.WriteLine($"  Construcción del modelo (pesos+gradientes+optimizer): {MemoryProfiler.FormatBytes(afterConstruct - beforeConstruct)}");
            Console.WriteLine($"  Forward (activaciones):                              {MemoryProfiler.FormatBytes(afterForward - afterConstruct)}");
            Console.WriteLine($"  Backward + clipping (buffers temporales):            {MemoryProfiler.FormatBytes(afterBackward - afterForward)}");
            Console.WriteLine($"  Update de pesos:                                     {MemoryProfiler.FormatBytes(afterUpdate - afterBackward)}");
            Console.WriteLine($"  Total acumulado tras un paso completo:               {MemoryProfiler.FormatBytes(afterUpdate - beforeConstruct)}");
            Console.WriteLine();

            GC.KeepAlive(model);
        }

        static long ForceCollectAndMeasure()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            return GC.GetTotalMemory(true);
        }

        static void PrintGpuBenchmarkResults(List<GpuBenchResult> results)
        {
            Console.WriteLine();
            Console.WriteLine($"{"Operación",-24} {"Forma",-24} {"CPU (ms)",10} {"GPU (ms)",10} {"Speedup",9} {"Gana",6}");
            Console.WriteLine(new string('-', 90));

            foreach (var r in results)
            {
                string gpuStr = r.GpuMs.HasValue ? r.GpuMs.Value.ToString("F3") : "n/a";
                string speedupStr = r.GpuMs.HasValue ? (r.CpuMs / r.GpuMs.Value).ToString("F2") + "x" : "n/a";
                string winner = r.GpuMs.HasValue ? (r.GpuMs.Value < r.CpuMs ? "GPU" : "CPU") : "CPU";

                Console.WriteLine($"{r.Op,-24} {r.Shape,-24} {r.CpuMs,10:F3} {gpuStr,10} {speedupStr,9} {winner,6}");
            }

            Console.WriteLine();
            Console.WriteLine("Umbral por operación (tamaño más chico probado donde la GPU ya gana):");

            foreach (var opGroup in results.GroupBy(r => r.Op))
            {
                var firstGpuWin = opGroup
                    .Where(r => r.GpuMs.HasValue && r.GpuMs.Value < r.CpuMs)
                    .OrderBy(r => r.ElementCount)
                    .FirstOrDefault();

                if (firstGpuWin != null)
                {
                    Console.WriteLine($"  {opGroup.Key,-24} -> GPU gana desde {firstGpuWin.Shape} ({firstGpuWin.ElementCount:N0} elementos)");
                }
                else
                {
                    Console.WriteLine($"  {opGroup.Key,-24} -> GPU no ganó en ningún tamaño probado; probá tamaños más grandes o dejá UseGpu=false para esta operación");
                }
            }

            Console.WriteLine();
            Console.WriteLine("Nota: estos tiempos incluyen la transferencia host<->device de cada llamada");
            Console.WriteLine("(las versiones *Auto/*CachedB del modelo evitan resubir pesos que no cambian,");
            Console.WriteLine("así que en entrenamiento real la GPU debería rendir mejor que lo medido acá");
            Console.WriteLine("para las operaciones que multiplican por una matriz de pesos).");
        }

        /// <summary>
        /// Carga los datos de entrenamiento desde archivos de texto plano (formato
        /// "Usuario:/Asistente:") ubicados en <paramref name="baseFolder"/>.
        /// El formato JSON indexado antiguo ya no se lee directamente aquí: conviértelo
        /// primero con <c>--convert</c> o <c>--convert-all</c>.
        /// </summary>
        static List<ConversationPair>? LoadTrainingData(string baseFolder)
        {
            try
            {
                if (!Directory.Exists(baseFolder))
                {
                    Console.WriteLine($"No se encontró la carpeta: {baseFolder}");
                    return null;
                }

                var txtFiles = Directory.GetFiles(baseFolder, "*.txt", SearchOption.AllDirectories);

                if (txtFiles.Length == 0)
                {
                    Console.WriteLine($"No se encontraron archivos .txt de entrenamiento en '{baseFolder}'.");

                    var jsonFiles = Directory.GetFiles(baseFolder, "*.json", SearchOption.AllDirectories);
                    if (jsonFiles.Length > 0)
                    {
                        Console.WriteLine($"Se encontraron {jsonFiles.Length} archivo(s) .json (formato antiguo), que ya no se cargan directamente.");
                        Console.WriteLine("Conviértelos primero a texto plano con:");
                        Console.WriteLine($"  dotnet run --project Neuraval.CLI -- --convert-all {baseFolder}");
                    }

                    return null;
                }

                Console.WriteLine($"Se encontraron {txtFiles.Length} archivo(s) de datos:");
                foreach (var file in txtFiles)
                {
                    Console.WriteLine($"  - {Path.GetFileName(file)}");
                }
                Console.WriteLine();

                var allConversations = new List<ConversationPair>();
                int loadedFiles = 0;

                foreach (var txtFile in txtFiles)
                {
                    try
                    {
                        Console.WriteLine($"Cargando: {Path.GetFileName(txtFile)}...");
                        var conversations = DatasetLoader.ParsePlainTextConversations(txtFile);

                        if (conversations.Count > 0)
                        {
                            allConversations.AddRange(conversations);
                            loadedFiles++;
                            Console.WriteLine($"  Cargados: {conversations.Count} pares de conversación");
                        }
                        else
                        {
                            Console.WriteLine($"  No se encontraron conversaciones válidas");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  Error al cargar {Path.GetFileName(txtFile)}: {ex.Message}");
                    }
                }

                Console.WriteLine();
                Console.WriteLine($"Resumen:");
                Console.WriteLine($"  Archivos cargados: {loadedFiles}/{txtFiles.Length}");
                Console.WriteLine($"  Pares de conversación totales: {allConversations.Count}");
                Console.WriteLine();

                return allConversations;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error al escanear la carpeta: {ex.Message}");
                return null;
            }
        }

        static async Task StartChat(TransformerChatBotService chatBot)
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("            ¡Chat iniciado!");
            Console.WriteLine("===========================================");
            Console.WriteLine();
            Console.WriteLine("Comandos:");
            Console.WriteLine("  - 'exit', 'quit' o 'salir' para terminar la conversación");
            Console.WriteLine("  - 'beam' para usar el modo beam search");
            Console.WriteLine("  - 'greedy' para usar el modo greedy (por defecto)");
            Console.WriteLine("  - 'temperature' para usar sampling con temperatura");
            Console.WriteLine("  - 'multiple' para generar varias respuestas");
            Console.WriteLine("  - 'info' para ver información del modelo");
            Console.WriteLine("  - 'stats' para ver estadísticas del sistema");
            Console.WriteLine();

            string generationMode = "greedy";
            double temperature = 0.7;

            while (true)
            {
                Console.Write("Tú: ");
                string input = Console.ReadLine() ?? "";

                if (string.IsNullOrWhiteSpace(input))
                {
                    continue;
                }

                input = input.Trim();
                string inputLower = input.ToLower();

                if (inputLower == "exit" || inputLower == "quit" || inputLower == "salir")
                {
                    Console.WriteLine();
                    Console.WriteLine("===========================================");
                    Console.WriteLine("        ¡Gracias por chatear!");
                    Console.WriteLine("===========================================");
                    break;
                }

                if (inputLower == "beam")
                {
                    generationMode = "beam";
                    Console.WriteLine("[Modo: Beam Search]");
                    Console.WriteLine();
                    continue;
                }

                if (inputLower == "greedy")
                {
                    generationMode = "greedy";
                    Console.WriteLine("[Modo: Greedy Decoding]");
                    Console.WriteLine();
                    continue;
                }

                if (inputLower == "temperature")
                {
                    generationMode = "temperature";
                    Console.Write("Ingresa el valor de temperatura (0.1-2.0, por defecto 0.7): ");
                    string tempInput = Console.ReadLine() ?? "";
                    if (double.TryParse(tempInput, out double temp) && temp > 0 && temp <= 2.0)
                    {
                        temperature = temp;
                    }
                    Console.WriteLine($"[Modo: Temperature Sampling (temp={temperature})]");
                    Console.WriteLine();
                    continue;
                }

                if (inputLower == "multiple")
                {
                    Console.WriteLine("\nGenerando varias respuestas...");
                    try
                    {
                        var responses = chatBot.GenerateMultipleCandidates(
                            input,
                            numCandidates: 5,
                            maxLength: 30,
                            temperature: 0.8
                        );

                        for (int i = 0; i < responses.Count; i++)
                        {
                            Console.WriteLine($"  {i + 1}. {responses[i]}");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error: {ex.Message}");
                    }
                    Console.WriteLine();
                    continue;
                }

                if (inputLower == "info")
                {
                    Console.WriteLine($"\n===========================================");
                    Console.WriteLine($"          Información del modelo");
                    Console.WriteLine($"===========================================");
                    Console.WriteLine($"  Tamaño de vocabulario: {chatBot.GetVocabularySize()}");
                    Console.WriteLine($"  Entrenado: {chatBot.IsTrained()}");
                    Console.WriteLine($"  Modo de generación: {generationMode}");
                    Console.WriteLine($"  Temperatura: {temperature}");
                    Console.WriteLine($"  Núcleos de CPU: {Environment.ProcessorCount}");
                    Console.WriteLine();
                    continue;
                }

                if (inputLower == "stats")
                {
                    SystemMonitor.PrintCurrentStats();
                    continue;
                }

                try
                {
                    var sw = Stopwatch.StartNew();
                    string response;

                    int maxLength = 10;

                    switch (generationMode)
                    {
                        case "beam":
                            response = chatBot.GenerateWithBeamSearch(
                                input,
                                maxLength: maxLength,
                                beamWidth: 5
                            );
                            break;

                        case "temperature":
                            response = chatBot.GenerateWithTemperature(
                                input,
                                maxLength: maxLength,
                                temperature: temperature
                            );
                            break;

                        case "greedy":
                        default:
                            var conversation = new List<ChatMessage> { new ChatMessage(ChatRole.User, input) };
                            var chatResponse = await chatBot.SendAsync(conversation);
                            response = chatResponse.Content;
                            break;
                    }

                    sw.Stop();
                    Console.WriteLine($"Bot: {response}");
                    Console.WriteLine($"     ({sw.ElapsedMilliseconds}ms)");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Bot: [Error: {ex.Message}]");
                }

                Console.WriteLine();
            }
        }
    }
}