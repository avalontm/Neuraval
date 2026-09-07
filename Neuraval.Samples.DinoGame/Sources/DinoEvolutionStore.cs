using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Neuraval.Samples.DinoGame.Sources
{
    /// <summary>
    /// Datos que se guardan en disco entre partidas: la generacion actual,
    /// el mejor fitness historico, el mejor genoma jamas visto (para no
    /// perderlo nunca, ni siquiera si una generacion "empeora" por mutacion)
    /// y una muestra de los genomas de elite de la ultima generacion (para
    /// poder reconstruir una poblacion diversa al reabrir el juego).
    /// </summary>
    public class DinoEvolutionSaveData
    {
        public int Generation { get; set; }
        public float BestFitnessEver { get; set; }
        public DinoGenome BestGenomeEver { get; set; }
        public List<DinoGenome> EliteGenomes { get; set; } = new List<DinoGenome>();
    }

    /// <summary>
    /// Persiste y reconstruye la evolucion de la poblacion de dinosaurios en
    /// disco (carpeta de datos de la aplicacion del usuario), para que al
    /// cerrar y volver a abrir el juego se conserve todo lo aprendido y la
    /// poblacion siga evolucionando desde donde se quedo, en vez de arrancar
    /// siempre desde cero con pesos aleatorios.
    /// </summary>
    public static class DinoEvolutionStore
    {
        private const int MaxEliteGenomesToSave = 60;

        private static readonly string SaveDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Neuraval.Samples.DinoGame");

        private static readonly string SaveFilePath = Path.Combine(SaveDirectory, "dino_evolution.json");

        public static DinoEvolutionSaveData Load()
        {
            try
            {
                if (!File.Exists(SaveFilePath))
                {
                    return null;
                }

                string json = File.ReadAllText(SaveFilePath);
                return JsonSerializer.Deserialize<DinoEvolutionSaveData>(json);
            }
            catch
            {
                // Si el archivo esta corrupto o de una version incompatible,
                // no debe impedir que el juego arranque: simplemente se
                // empieza una poblacion nueva, como si fuera la primera vez.
                return null;
            }
        }

        public static void Save(DinoEvolutionSaveData data)
        {
            try
            {
                Directory.CreateDirectory(SaveDirectory);
                string json = JsonSerializer.Serialize(data);

                // Escritura atomica: primero a un archivo temporal y luego
                // se reemplaza, para no dejar un JSON a medio escribir si el
                // juego se cierra justo durante el guardado.
                string tempPath = SaveFilePath + ".tmp";
                File.WriteAllText(tempPath, json);
                File.Copy(tempPath, SaveFilePath, overwrite: true);
                File.Delete(tempPath);
            }
            catch
            {
                // Persistir la evolucion es "best effort": si falla (por
                // permisos, disco lleno, etc.) el juego debe poder seguir
                // jugandose con normalidad en memoria.
            }
        }

        /// <summary>
        /// Construye los datos a guardar a partir de la generacion que
        /// acaba de terminar: conserva el mejor genoma historico (aunque la
        /// generacion actual haya sido peor) y guarda una muestra de los
        /// mejores genomas de esta generacion para poder repoblar al reabrir.
        /// </summary>
        public static DinoEvolutionSaveData BuildSaveData(
            IReadOnlyList<(DinoGenome Genome, float Fitness)> rankedDescending,
            int generation,
            float bestFitnessEver,
            DinoGenome bestGenomeEver)
        {
            float bestThisGeneration = rankedDescending.Count > 0 ? rankedDescending[0].Fitness : 0f;

            if (bestThisGeneration > bestFitnessEver || bestGenomeEver == null)
            {
                bestFitnessEver = bestThisGeneration;
                bestGenomeEver = rankedDescending[0].Genome;
                bestGenomeEver.Fitness = bestThisGeneration;
            }

            var eliteGenomes = rankedDescending
                .Take(MaxEliteGenomesToSave)
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

        /// <summary>
        /// Reconstruye una poblacion de <paramref name="populationSize"/>
        /// cerebros a partir de datos guardados: el mejor cerebro historico
        /// se conserva intacto (garantiza que nunca se pierde lo aprendido),
        /// y el resto se rellena mezclando clones mutados de los genomas de
        /// elite guardados con algunos cerebros totalmente nuevos para
        /// mantener diversidad genetica.
        /// </summary>
        public static List<NeuralNetwork> RebuildPopulation(
            DinoEvolutionSaveData data,
            int populationSize,
            Random random,
            float mutationRate,
            float mutationStrength)
        {
            var brains = new List<NeuralNetwork>(populationSize);

            if (data.BestGenomeEver != null)
            {
                // El mejor de la historia siempre sobrevive sin mutar.
                brains.Add(NeuralNetwork.FromGenome(data.BestGenomeEver));
            }

            var seedGenomes = data.EliteGenomes.Count > 0
                ? data.EliteGenomes
                : (data.BestGenomeEver != null ? new List<DinoGenome> { data.BestGenomeEver } : new List<DinoGenome>());

            if (seedGenomes.Count > 0)
            {
                // ~80% de la poblacion se rellena con descendencia mutada de
                // los genomas guardados, para seguir explorando a partir de
                // lo ya aprendido.
                int toBreed = (int)(populationSize * 0.8f);

                while (brains.Count < toBreed)
                {
                    var parentGenome = seedGenomes[random.Next(seedGenomes.Count)];
                    var parent = NeuralNetwork.FromGenome(parentGenome);
                    brains.Add(parent.CloneWithMutation(random, mutationRate, mutationStrength));
                }
            }

            // El resto se completa con cerebros nuevos aleatorios, para no
            // estancar la evolucion en un unico linaje.
            while (brains.Count < populationSize)
            {
                brains.Add(new NeuralNetwork());
            }

            return brains;
        }
    }
}
