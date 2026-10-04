using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Neuraval.Samples.DinoGame.Sources
{
    public class DinoEvolutionSaveData
    {
        public int Generation { get; set; }
        public float BestFitnessEver { get; set; }
        public DinoGenome BestGenomeEver { get; set; }
        public List<DinoGenome> EliteGenomes { get; set; } = new List<DinoGenome>();
    }

    public static class DinoEvolutionStore
    {
        private static readonly string SaveDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Neuraval.Samples.DinoGame");

        private static readonly string SaveFilePath = Path.Combine(SaveDirectory, "dino_evolution.json");

        public static DinoEvolutionSaveData Load()
        {
            if (!File.Exists(SaveFilePath))
            {
                return null;
            }

            string json;

            try
            {
                json = File.ReadAllText(SaveFilePath);
            }
            catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException)
            {
                SampleDiagnostics.Warn("dino_evolution.json could not be read", exception);
                return null;
            }

            try
            {
                return JsonSerializer.Deserialize<DinoEvolutionSaveData>(json);
            }
            catch (JsonException exception)
            {
                SampleDiagnostics.Warn("dino_evolution.json is not valid JSON; a copy is being preserved before the next save overwrites it", exception);
                PreserveCorruptSave();
                return null;
            }
        }

        private static void PreserveCorruptSave()
        {
            string backupPath = Path.Combine(SaveDirectory, $"dino_evolution.corrupt-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json");

            try
            {
                File.Copy(SaveFilePath, backupPath, overwrite: true);
                SampleDiagnostics.Warn($"corrupt evolution save preserved at {backupPath}", null);
            }
            catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException)
            {
                SampleDiagnostics.Warn($"corrupt evolution save could not be preserved at {backupPath}", exception);
            }
        }

        public static void Save(DinoEvolutionSaveData data)
        {
            try
            {
                Directory.CreateDirectory(SaveDirectory);
                string json = JsonSerializer.Serialize(data);

                string tempPath = SaveFilePath + ".tmp";
                File.WriteAllText(tempPath, json);
                File.Move(tempPath, SaveFilePath, overwrite: true);
            }
            catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException)
            {
                SampleDiagnostics.Warn("dino_evolution.json could not be written", exception);
            }
        }

        public static bool Delete()
        {
            try
            {
                if (File.Exists(SaveFilePath))
                {
                    File.Delete(SaveFilePath);
                }

                string tempPath = SaveFilePath + ".tmp";

                if (File.Exists(tempPath))
                {
                    File.Delete(tempPath);
                }

                return true;
            }
            catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException)
            {
                SampleDiagnostics.Warn("dino_evolution.json could not be deleted", exception);
                return false;
            }
        }

        public static DinoEvolutionSaveData BuildSaveData(
            IReadOnlyList<(DinoGenome Genome, float Fitness)> rankedDescending,
            int generation,
            float bestFitnessEver,
            DinoGenome bestGenomeEver,
            int maxEliteGenomesToSave)
        {
            float bestThisGeneration = rankedDescending.Count > 0 ? rankedDescending[0].Fitness : 0f;

            if (bestThisGeneration > bestFitnessEver || bestGenomeEver == null)
            {
                bestFitnessEver = bestThisGeneration;
                bestGenomeEver = rankedDescending[0].Genome;
                bestGenomeEver.Fitness = bestThisGeneration;
            }

            var eliteGenomes = rankedDescending
                .Take(maxEliteGenomesToSave)
                .Select(pair =>
                {
                    pair.Genome.Fitness = pair.Fitness;
                    return pair.Genome;
                })
                .ToList();

            return new DinoEvolutionSaveData
            {
                Generation = generation,
                BestFitnessEver = bestFitnessEver,
                BestGenomeEver = bestGenomeEver,
                EliteGenomes = eliteGenomes
            };
        }

        public static List<NeuralNetwork> RebuildPopulation(
            DinoEvolutionSaveData data,
            int populationSize,
            Random random,
            float mutationRate,
            float mutationStrength,
            float randomInjectionFraction)
        {
            var brains = new List<NeuralNetwork>(populationSize);

            if (data.BestGenomeEver != null)
            {
                brains.Add(NeuralNetwork.FromGenome(data.BestGenomeEver));
            }

            var seedGenomes = data.EliteGenomes.Count > 0
                ? data.EliteGenomes
                : (data.BestGenomeEver != null ? new List<DinoGenome> { data.BestGenomeEver } : new List<DinoGenome>());

            if (seedGenomes.Count > 0)
            {
                int toBreed = (int)(populationSize * (1f - randomInjectionFraction));

                while (brains.Count < toBreed)
                {
                    var parentGenome = seedGenomes[random.Next(seedGenomes.Count)];
                    var parent = NeuralNetwork.FromGenome(parentGenome);
                    brains.Add(parent.CloneWithMutation(random, mutationRate, mutationStrength));
                }
            }

            while (brains.Count < populationSize)
            {
                brains.Add(new NeuralNetwork());
            }

            return brains;
        }
    }
}
