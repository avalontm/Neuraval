using System.Net;
using Neuraval.Evolution;
using Neuraval.Evolution.Neat;

namespace Neuraval.Evolution.MarioBridge
{
    internal static class Program
    {
        private const string Address = "127.0.0.1";
        private const int Port = 8766;
        private const int DefaultPopulationSize = 16;
        private const int MaxStepsPerEpisode = 1200;
        private const string DefaultCheckpointPath = "checkpoints/mario_checkpoint.navm";
        private const string DefaultModelPath = "checkpoints/mario_best.navm";
        private const string DefaultCapturePath = "checkpoints/mario_dataset.navm";
        private const string DefaultPolicyPath = "checkpoints/mario_policy.navm";

        private static readonly string[] DefaultLevels = { "Nivel 0 (DP1.state)" };

        private static void Main(string[] args)
        {
            var imitateDataset = GetOption(args, "--imitate");
            if (imitateDataset != null)
            {
                var outputPath = GetOptionOr(args, "--imitate-out", DefaultPolicyPath);
                var epochs = TryGetInt(GetOption(args, "--epochs"), 20);
                RunImitationMode(imitateDataset, outputPath, epochs);
                return;
            }

            var capturePath = GetOption(args, "--capture");
            if (capturePath != null)
            {
                RunCaptureMode(capturePath);
                return;
            }

            var populationSize = TryGetInt(GetOption(args, "--population"), DefaultPopulationSize);
            RunTrainingMode(args, populationSize);
        }

        private static void RunCaptureMode(string capturePath)
        {
            using var connection = new SnesBridgeConnection(IPAddress.Parse(Address), Port);
            var stopRequested = false;

            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                stopRequested = true;
                Console.WriteLine("\nDeteniendo captura y guardando dataset...");
            };

            Console.WriteLine($"Esperando conexion de BizHawk en {Address}:{Port}...");
            connection.WaitForBizHawk();
            Console.WriteLine("BizHawk conectado. Juega con el teclado: cada frame se graba junto con tus botones.");
            Console.WriteLine("Muertes, fin de nivel y el reset manual (tecla Insert) reinician el nivel y cierran el episodio.");
            Console.WriteLine("Ctrl+C para cerrar el dataset y salir.");

            using var recorder = new MarioDatasetRecorder(capturePath);
            var state = connection.ReceiveState();
            int? previousX = null;
            int? previousCoins = null;
            int? previousPowerup = null;
            var frames = 0;
            var episodes = 0;

            while (!stopRequested)
            {
                var action = MarioControllerEncoder.Decode(state.Controller1, state.Controller2);

                if (state.IsDead || state.IsLevelComplete || state.ManualResetRequested)
                {
                    episodes++;
                    recorder.Append(state, action, 0f, done: true);

                    var reason = state.IsLevelComplete ? "nivel completado" : state.IsDead ? "muerte" : "reset manual";
                    Console.WriteLine($"    Episodio {episodes} terminado ({reason}): marioX final {state.MarioX}.");

                    connection.SendCapture(state.LevelIndex);
                    previousX = null;
                    previousCoins = null;
                    previousPowerup = null;
                }
                else
                {
                    var reward = previousX.HasValue ? state.MarioX - previousX.Value : 0f;

                    if (previousCoins.HasValue && state.Coins > previousCoins.Value)
                    {
                        reward += (state.Coins - previousCoins.Value) * SnesEnvironment.CoinReward;
                    }

                    if (previousPowerup.HasValue && state.PowerupLevel != previousPowerup.Value)
                    {
                        var powerupDelta = state.PowerupLevel - previousPowerup.Value;
                        reward += powerupDelta > 0
                            ? powerupDelta * SnesEnvironment.PowerupGainReward
                            : powerupDelta * SnesEnvironment.PowerupLossPenalty;
                    }

                    previousX = state.MarioX;
                    previousCoins = state.Coins;
                    previousPowerup = state.PowerupLevel;
                    recorder.Append(state, action, reward, done: false);
                    frames++;

                    connection.SendAction(SnesAction.None);
                }

                state = connection.ReceiveState();

                if (frames > 0 && frames % 600 == 0)
                {
                    Console.WriteLine($"    Capturando... {frames} frames, {episodes} episodios.");
                }
            }

            recorder.Complete();
        }

        private static void RunImitationMode(string datasetPath, string outputPath, int epochs)
        {
            Console.WriteLine($"Cargando dataset: {Path.GetFullPath(datasetPath)}");
            var dataset = MarioDatasetLoader.Load(datasetPath);

            if (dataset == null)
            {
                Console.WriteLine("No se pudo leer el dataset; se cancela el entrenamiento de imitacion.");
                return;
            }

            Console.WriteLine($"Dataset: {dataset.Samples.Count} muestras, {dataset.InputCount} entradas, {dataset.OutputCount} salidas.");

            var trainer = new MarioImitationTrainer(dataset, epochs);
            var policy = trainer.Train(new Random(1234));
            policy.Save(outputPath);

            Console.WriteLine($"Politica guardada: {Path.GetFullPath(outputPath)}");
            Console.WriteLine("Usa --seed-policy <archivo> al volver a entrenar por NEAT para arrancar desde esta politica.");
        }

        private static void RunTrainingMode(string[] args, int populationSize)
        {
            var resetRequested = args.Any(arg => arg.Equals("--reset", StringComparison.OrdinalIgnoreCase));
            var checkpointPath = GetOption(args, "--checkpoint");
            var seedPolicyPath = GetOption(args, "--seed-policy");
            var exportModelPath = GetOptionOr(args, "--export-model", DefaultModelPath);
            var levels = ResolveLevels(args);

            if (checkpointPath != null)
            {
                MarioCheckpointStore.SaveFilePath = checkpointPath;
            }

            Console.WriteLine($"Archivo de entrenamiento: {Path.GetFullPath(MarioCheckpointStore.SaveFilePath)}");

            if (resetRequested)
            {
                MarioCheckpointStore.Delete();
                Console.WriteLine("Progreso anterior eliminado, arrancando desde cero.");
            }

            using var connection = new SnesBridgeConnection(IPAddress.Parse(Address), Port);
            var stopRequested = false;

            Console.CancelKeyPress += (_, e) =>
            {
                if (stopRequested)
                {
                    Console.WriteLine("\nForzando cierre inmediato. El progreso de la generacion en curso puede perderse (el ultimo checkpoint guardado sigue intacto).");
                    return;
                }

                e.Cancel = true;
                stopRequested = true;
                Console.WriteLine("\nSaliendo despues de esta generacion... puede tardar si esta esperando datos de BizHawk. Presiona Ctrl+C de nuevo para forzar el cierre inmediato.");
            };

            Console.WriteLine($"Esperando conexion de BizHawk en {Address}:{Port}...");
            connection.WaitForBizHawk();
            Console.WriteLine("BizHawk conectado.");

            var environment = new SnesEnvironment(connection);
            var random = new Random(1234);
            var tracker = new NeatInnovationTracker(MarioAgent.InputCount + MarioAgent.OutputCount + 1);

            var checkpoint = MarioCheckpointStore.Load();

            MarioAgent[] initialAgents;
            int startingGeneration;
            var bestFitnessEver = 0f;
            NeatGenome? bestGenomeEver = null;

            if (checkpoint != null && checkpoint.Genomes.Count > 0)
            {
                var maxNodeId = checkpoint.Genomes.SelectMany(genome => genome.Nodes).Max(node => node.Id);
                var maxInnovation = checkpoint.Genomes
                    .SelectMany(genome => genome.Connections)
                    .Select(gene => (int?)gene.Innovation)
                    .DefaultIfEmpty()
                    .Max() ?? -1;

                tracker.FastForwardTo(maxNodeId + 1, maxInnovation + 1);

                initialAgents = checkpoint.Genomes.Select(genome => new MarioAgent(genome)).ToArray();
                startingGeneration = checkpoint.Generation;
                bestFitnessEver = checkpoint.BestFitnessEver;
                bestGenomeEver = checkpoint.BestGenomeEver;

                Console.WriteLine($"Progreso anterior encontrado: reanudando desde la generacion {startingGeneration}.");
            }
            else
            {
                var seedGenome = TryLoadSeedGenome(seedPolicyPath, tracker);

                initialAgents = new MarioAgent[populationSize];
                for (var i = 0; i < populationSize; i++)
                {
                    initialAgents[i] = seedGenome != null && i == 0
                        ? new MarioAgent(seedGenome.Clone())
                        : MarioAgent.CreateRandom(random, tracker);
                }

                startingGeneration = 0;
            }

            var options = new NeatEvolutionOptions();
            var strategy = new NeatEvolutionStrategy<MarioAgent>(random, tracker, options, genome => new MarioAgent(genome));

            var population = new Population<MarioAgent, SnesState, SnesAction>(
                initialAgents,
                () => environment,
                new MarioFitnessEvaluator(MaxStepsPerEpisode),
                strategy);

            Console.WriteLine("Neuraval.Evolution.MarioBridge");
            Console.WriteLine("Generacion | Especies | Fitness promedio | Mejor fitness");

            var generation = startingGeneration;

            while (!stopRequested)
            {
                generation++;

                environment.LevelIndex = levels.Length == 0 ? 0 : generation % levels.Length;
                var levelLabel = environment.LevelIndex < levels.Length
                    ? levels[environment.LevelIndex]
                    : environment.LevelIndex.ToString();
                Console.WriteLine($"-- Generacion {generation}: entrenando en \"{levelLabel}\" --");

                var scores = population.EvaluateGeneration();
                var bestThisGeneration = scores.Count == 0 ? 0f : scores.Max();

                if (bestThisGeneration > bestFitnessEver || bestGenomeEver == null)
                {
                    var bestIndex = scores.ToList().IndexOf(bestThisGeneration);
                    bestFitnessEver = bestThisGeneration;
                    bestGenomeEver = population.Agents[bestIndex].Genome.Clone();
                }

                strategy.GlobalBestGenome = bestGenomeEver;
                population.Advance();
                Console.WriteLine($"{generation,10} | {strategy.LastSpeciesCount,8} | {population.AverageFitness,17:F2} | {population.BestFitness,13:F2}");

                MarioCheckpointStore.Save(new MarioCheckpoint
                {
                    Generation = generation,
                    BestFitnessEver = bestFitnessEver,
                    BestGenomeEver = bestGenomeEver,
                    Genomes = population.Agents.Select(agent => agent.Genome).ToList()
                });

                if (bestGenomeEver != null)
                {
                    MarioNeatModelStore.Save(exportModelPath, bestGenomeEver);
                }
            }

            Console.WriteLine("Entrenamiento detenido por el usuario. Progreso guardado.");
        }

        private static NeatGenome? TryLoadSeedGenome(string? seedPolicyPath, NeatInnovationTracker tracker)
        {
            if (seedPolicyPath == null)
            {
                return null;
            }

            var policy = MarioPolicyNetwork.Load(seedPolicyPath);
            if (policy == null)
            {
                Console.WriteLine($"No se pudo cargar la politica semilla en {Path.GetFullPath(seedPolicyPath)}; se arranca con genomas random.");
                return null;
            }

            Console.WriteLine($"Sembrando poblacion con politica imitada: {Path.GetFullPath(seedPolicyPath)}.");

            var genome = policy.AsGenome(tracker);
            var maxNodeId = genome.Nodes.Max(node => node.Id);
            var maxInnovation = genome.Connections.Count - 1;
            tracker.FastForwardTo(maxNodeId + 1, maxInnovation + 1);
            return genome;
        }

        private static string[] ResolveLevels(string[] args)
        {
            var levels = new List<string>();

            for (var i = 0; i + 1 < args.Length; i++)
            {
                if (args[i].Equals("--level", StringComparison.OrdinalIgnoreCase))
                {
                    levels.Add(args[i + 1]);
                    i++;
                }
            }

            return levels.Count > 0 ? levels.ToArray() : (string[])DefaultLevels.Clone();
        }

        private static string? GetOption(string[] args, string name)
        {
            var index = Array.FindIndex(args, arg => arg.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (index >= 0 && index + 1 < args.Length)
            {
                return args[index + 1];
            }

            return null;
        }

        private static string GetOptionOr(string[] args, string name, string fallback)
        {
            return GetOption(args, name) ?? fallback;
        }

        private static int TryGetInt(string? value, int fallback)
        {
            return int.TryParse(value, out var parsed) ? parsed : fallback;
        }
    }
}