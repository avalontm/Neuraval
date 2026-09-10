using System.Text.Json;

namespace Neuraval.CLI
{
    public class TrainingSettings
    {
        public string? Preset { get; set; }
        public int EmbeddingDim { get; set; } = 128;
        public int NumLayers { get; set; } = 4;
        public int NumHeads { get; set; } = 4;
        public int FeedforwardDim { get; set; } = 512;
        public int MaxSequenceLength { get; set; } = 128;
        public double Dropout { get; set; } = 0.1;
        public int VocabSize { get; set; } = 10000;

        public int BatchSize { get; set; } = 32;
        public int GradientAccumulationSteps { get; set; } = 1;
        public double LearningRate { get; set; } = 0.001;
        public int Epochs { get; set; } = 500;
        public double ValidationSplit { get; set; } = 0.2;
        public int Patience { get; set; } = 20;

        public int CheckpointEveryEpochs { get; set; } = 5;
        public int NumThreads { get; set; } = -1;
        public bool UseGpu { get; set; } = false;

        public string ModelPath { get; set; } = "SavedModel";
        public string DataFolder { get; set; } = "Data";

        /// <summary>
        /// Rank de los adaptadores LoRA (Fase 5.5). 0 (default) = LoRA
        /// deshabilitado, entrenamiento full fine-tuning como siempre. Un
        /// valor &gt; 0 habilita LoRA en las cuatro proyecciones de atención
        /// de cada bloque y congela el resto del modelo (embedding, FFN,
        /// LayerNorms, norma final y bias de salida): solo se entrenan los
        /// adaptadores A/B.
        /// </summary>
        public int LoraRank { get; set; } = 0;

        /// <summary>Alpha de LoRA (solo aplica si <see cref="LoraRank"/> &gt; 0).</summary>
        public double LoraAlpha { get; set; } = 16.0;

        public static TrainingSettings Load(string filepath, string[] args)
        {
            var settings = File.Exists(filepath)
                ? LoadFromFile(filepath)
                : new TrainingSettings();

            if (!File.Exists(filepath))
            {
                Save(settings, filepath);
                Console.WriteLine($"No se encontró {filepath}, se creó uno con los valores por defecto.");
                Console.WriteLine();
            }

            ApplyArgOverrides(settings, args);

            return settings;
        }

        private static TrainingSettings LoadFromFile(string filepath)
        {
            try
            {
                var json = File.ReadAllText(filepath);
                var settings = JsonSerializer.Deserialize<TrainingSettings>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                return settings ?? new TrainingSettings();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"No se pudo leer {filepath} ({ex.Message}), usando valores por defecto.");
                return new TrainingSettings();
            }
        }

        private static void Save(TrainingSettings settings, string filepath)
        {
            var json = JsonSerializer.Serialize(settings, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(filepath, json);
        }

        private static readonly Dictionary<string, ModelPreset> Presets = new(StringComparer.OrdinalIgnoreCase)
        {
            ["nano"] = new ModelPreset(EmbeddingDim: 128, NumLayers: 2, NumHeads: 4, FeedforwardDim: 512, MaxSequenceLength: 256),
            ["small"] = new ModelPreset(EmbeddingDim: 256, NumLayers: 6, NumHeads: 8, FeedforwardDim: 1024, MaxSequenceLength: 512),
            ["medium"] = new ModelPreset(EmbeddingDim: 512, NumLayers: 8, NumHeads: 8, FeedforwardDim: 2048, MaxSequenceLength: 1024),
            ["large"] = new ModelPreset(EmbeddingDim: 768, NumLayers: 12, NumHeads: 12, FeedforwardDim: 3072, MaxSequenceLength: 2048),
        };

        private static void ApplyArgOverrides(TrainingSettings settings, string[] args)
        {
            ApplyPresetIfPresent(settings, args);

            for (int i = 0; i < args.Length - 1; i++)
            {
                var key = args[i].TrimStart('-').ToLowerInvariant();
                var value = args[i + 1];

                switch (key)
                {
                    case "preset": settings.Preset = value; break;
                    case "embeddingdim": settings.EmbeddingDim = int.Parse(value); break;
                    case "numlayers": settings.NumLayers = int.Parse(value); break;
                    case "numheads": settings.NumHeads = int.Parse(value); break;
                    case "feedforwarddim": settings.FeedforwardDim = int.Parse(value); break;
                    case "maxsequencelength": settings.MaxSequenceLength = int.Parse(value); break;
                    case "dropout": settings.Dropout = double.Parse(value); break;
                    case "vocabsize": settings.VocabSize = int.Parse(value); break;
                    case "batchsize": settings.BatchSize = int.Parse(value); break;
                    case "gradientaccumulationsteps": settings.GradientAccumulationSteps = int.Parse(value); break;
                    case "learningrate": settings.LearningRate = double.Parse(value); break;
                    case "epochs": settings.Epochs = int.Parse(value); break;
                    case "validationsplit": settings.ValidationSplit = double.Parse(value); break;
                    case "patience": settings.Patience = int.Parse(value); break;
                    case "checkpointeveryepochs": settings.CheckpointEveryEpochs = int.Parse(value); break;
                    case "numthreads": settings.NumThreads = int.Parse(value); break;
                    case "usegpu": settings.UseGpu = bool.Parse(value); break;
                    case "modelpath": settings.ModelPath = value; break;
                    case "datafolder": settings.DataFolder = value; break;
                    case "lorarank": settings.LoraRank = int.Parse(value); break;
                    case "loraalpha": settings.LoraAlpha = double.Parse(value); break;
                }
            }
        }

        private static void ApplyPresetIfPresent(TrainingSettings settings, string[] args)
        {
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (string.Equals(args[i].TrimStart('-'), "preset", StringComparison.OrdinalIgnoreCase))
                {
                    ApplyPreset(settings, args[i + 1]);
                    return;
                }
            }
        }

        private static void ApplyPreset(TrainingSettings settings, string presetName)
        {
            if (!Presets.TryGetValue(presetName, out var preset))
            {
                Console.WriteLine($"Preset desconocido: '{presetName}'. Presets disponibles: {string.Join(", ", Presets.Keys)}");
                Console.WriteLine();
                return;
            }

            settings.Preset = presetName;
            settings.EmbeddingDim = preset.EmbeddingDim;
            settings.NumLayers = preset.NumLayers;
            settings.NumHeads = preset.NumHeads;
            settings.FeedforwardDim = preset.FeedforwardDim;
            settings.MaxSequenceLength = preset.MaxSequenceLength;
        }

        public void Print()
        {
            Console.WriteLine("Configuración de entrenamiento:");
            Console.WriteLine($"  Preset: {Preset ?? "(ninguno, valores manuales/por defecto)"}");
            Console.WriteLine($"  Modelo: embeddingDim={EmbeddingDim}, numLayers={NumLayers}, numHeads={NumHeads}, feedforwardDim={FeedforwardDim}, maxSequenceLength={MaxSequenceLength}, dropout={Dropout}, vocabSize={VocabSize}");
            Console.WriteLine($"  Entrenamiento: epochs={Epochs}, batchSize={BatchSize}, gradientAccumulationSteps={GradientAccumulationSteps} (batch efectivo={BatchSize * GradientAccumulationSteps}), learningRate={LearningRate}, validationSplit={ValidationSplit}, patience={Patience}");
            Console.WriteLine($"  Checkpoints cada {CheckpointEveryEpochs} épocas, numThreads={(NumThreads > 0 ? NumThreads.ToString() : "auto")}, useGpu={UseGpu}");
            Console.WriteLine($"  ModelPath={ModelPath}, DataFolder={DataFolder}");
            if (LoraRank > 0)
            {
                Console.WriteLine($"  LoRA habilitado: rank={LoraRank}, alpha={LoraAlpha} (base congelada, solo se entrenan los adaptadores)");
            }
            PrintAttentionMemoryEstimate();
            Console.WriteLine();
        }

        /// <summary>
        /// El path de entrenamiento (training=true) no usa Flash Attention -a propósito, porque
        /// el backward necesita los pesos de atención completos-, así que cachea una matriz
        /// batchSize x numHeads x seqLen x seqLen por capa mientras dura el forward+backward del
        /// batch. Ese es el término dominante de memoria al subir MaxSequenceLength, y crece con
        /// el cuadrado del contexto. Esto solo estima y avisa, no cambia ningún comportamiento.
        /// </summary>
        private void PrintAttentionMemoryEstimate()
        {
            const long BytesPerFloat = 4;
            const long WarningThresholdBytes = 4L * 1024 * 1024 * 1024;

            long attentionWeightsBytes = (long)NumLayers * BatchSize * NumHeads
                * MaxSequenceLength * MaxSequenceLength * BytesPerFloat;

            double attentionWeightsMb = attentionWeightsBytes / (1024.0 * 1024.0);

            Console.WriteLine($"  Estimación de memoria de atención en training (solo pesos de atención, crece con maxSequenceLength^2): ~{attentionWeightsMb:F0} MB");

            if (attentionWeightsBytes > WarningThresholdBytes)
            {
                Console.WriteLine("  ADVERTENCIA: esta estimación supera los 4 GB. Si te quedás sin memoria, bajá BatchSize");
                Console.WriteLine("  y compensá con GradientAccumulationSteps (mismo batch efectivo, memoria pico menor),");
                Console.WriteLine("  o bajá MaxSequenceLength / NumLayers.");
            }
        }
    }

    internal readonly record struct ModelPreset(int EmbeddingDim, int NumLayers, int NumHeads, int FeedforwardDim, int MaxSequenceLength);
}
