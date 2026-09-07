using System;
using System.IO;
using System.Linq;
using System.Net;
using Neuraval.Evolution;
using Neuraval.Evolution.Neat;

namespace Neuraval.Evolution.MarioBridge
{
    internal static class Program
    {
        private const string Address = "127.0.0.1";
        private const int Port = 8766;
        private const int PopulationSize = 5;
        private const int MaxStepsPerEpisode = 1200;

        // Nombres solo para el log de consola -- no afectan el entrenamiento
        // en nada (el indice numerico es lo que se manda a Lua, ver
        // environment.LevelIndex mas abajo). Puse "DP1" tal cual el nombre
        // del archivo que ya tenias (D:/_CODE_/BizHawk/DP1.state); no se a
        // ciencia cierta que nivel es, cambialo por el nombre real si lo
        // sabes, o dejalo generico. Tiene que tener la MISMA cantidad de
        // entradas que SAVESTATE_FILES en mario_bridge.lua, en el mismo
        // orden -- son dos archivos separados que no se sincronizan solos.
        private static readonly string[] MarioLevels = { "Nivel 0 (DP1.state)" };

        private static void Main(string[] args)
        {
            var resetRequested = args.Any(arg => arg.Equals("--reset", StringComparison.OrdinalIgnoreCase));

            var checkpointIndex = Array.FindIndex(args, arg => arg.Equals("--checkpoint", StringComparison.OrdinalIgnoreCase));
            if (checkpointIndex >= 0 && checkpointIndex + 1 < args.Length)
            {
                MarioCheckpointStore.SaveFilePath = args[checkpointIndex + 1];
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
                    // Segundo Ctrl+C: el usuario ya esperó y quiere salir ya.
                    // No cancelamos el evento, asi el proceso termina de una.
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
            NeatGenome bestGenomeEver = null;

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
                initialAgents = new MarioAgent[PopulationSize];
                for (var i = 0; i < PopulationSize; i++)
                {
                    initialAgents[i] = MarioAgent.CreateRandom(random, tracker);
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

                // Rotacion de niveles: TODA la poblacion de una generacion
                // juega el mismo nivel (para que el fitness sea comparable
                // entre genomas dentro de esa generacion), pero el nivel
                // cambia de una generacion a la siguiente en orden (round
                // robin). Con un solo nivel configurado esto no cambia nada
                // respecto a antes -- agregar mas savestates en Lua y mas
                // nombres en MarioLevels activa la rotacion sin tocar mas
                // codigo.
                environment.LevelIndex = MarioLevels.Length == 0 ? 0 : generation % MarioLevels.Length;
                var levelLabel = environment.LevelIndex < MarioLevels.Length
                    ? MarioLevels[environment.LevelIndex]
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

                // Asegura que el mejor genoma de toda la corrida siempre
                // sobreviva a la proxima generacion, sin importar como
                // termine especiandose la poblacion (ver comentario en
                // NeatEvolutionStrategy.GlobalBestGenome).
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
            }

            Console.WriteLine("Entrenamiento detenido por el usuario. Progreso guardado.");
        }
    }
}
