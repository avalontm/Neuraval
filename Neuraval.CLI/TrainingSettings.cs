using System.Text.Json;

namespace Neuraval.CLI
{
    public class TrainingSettings
    {
        public int EmbeddingDim { get; set; } = 128;
        public int NumLayers { get; set; } = 4;
        public int NumHeads { get; set; } = 4;
        public int FeedforwardDim { get; set; } = 512;
        public int MaxSequenceLength { get; set; } = 128;
        public double Dropout { get; set; } = 0.1;

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

        private static void ApplyArgOverrides(TrainingSettings settings, string[] args)
        {
            for (int i = 0; i < args.Length - 1; i++)
            {
                var key = args[i].TrimStart('-').ToLowerInvariant();
                var value = args[i + 1];

                switch (key)
                {
                    case "embeddingdim": settings.EmbeddingDim = int.Parse(value); break;
                    case "numlayers": settings.NumLayers = int.Parse(value); break;
                    case "numheads": settings.NumHeads = int.Parse(value); break;
                    case "feedforwarddim": settings.FeedforwardDim = int.Parse(value); break;
                    case "maxsequencelength": settings.MaxSequenceLength = int.Parse(value); break;
                    case "dropout": settings.Dropout = double.Parse(value); break;
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
                }
            }
        }

        public void Print()
        {
            Console.WriteLine("Configuración de entrenamiento:");
            Console.WriteLine($"  Modelo: embeddingDim={EmbeddingDim}, numLayers={NumLayers}, numHeads={NumHeads}, feedforwardDim={FeedforwardDim}, maxSequenceLength={MaxSequenceLength}, dropout={Dropout}");
            Console.WriteLine($"  Entrenamiento: epochs={Epochs}, batchSize={BatchSize}, gradientAccumulationSteps={GradientAccumulationSteps} (batch efectivo={BatchSize * GradientAccumulationSteps}), learningRate={LearningRate}, validationSplit={ValidationSplit}, patience={Patience}");
            Console.WriteLine($"  Checkpoints cada {CheckpointEveryEpochs} épocas, numThreads={(NumThreads > 0 ? NumThreads.ToString() : "auto")}, useGpu={UseGpu}");
            Console.WriteLine($"  ModelPath={ModelPath}, DataFolder={DataFolder}");
            Console.WriteLine();
        }
    }
}
