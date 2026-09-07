using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Evolution.Neat
{
    public sealed class NeatEvolutionStrategy<TAgent> : IEvolutionStrategy<TAgent> where TAgent : INeatAgent<TAgent>
    {
        private readonly Random _random;
        private readonly NeatInnovationTracker _tracker;
        private readonly NeatEvolutionOptions _options;
        private readonly Func<NeatGenome, TAgent> _agentFactory;

        public int LastSpeciesCount { get; private set; }

        public NeatEvolutionStrategy(Random random, NeatInnovationTracker tracker, NeatEvolutionOptions options, Func<NeatGenome, TAgent> agentFactory)
        {
            _random = random;
            _tracker = tracker;
            _options = options;
            _agentFactory = agentFactory;
        }

        public IReadOnlyList<TAgent> NextGeneration(IReadOnlyList<TAgent> currentGeneration, IReadOnlyList<float> fitnessScores)
        {
            _tracker.ResetGenerationCache();

            var population = currentGeneration
                .Select((agent, index) => (Genome: agent.Genome, Fitness: fitnessScores[index]))
                .ToList();

            var speciesGroups = Speciate(population);
            LastSpeciesCount = speciesGroups.Count;

            var adjustedFitnessSums = speciesGroups
                .Select(group => group.Sum(individual => individual.Fitness / group.Count))
                .ToList();

            var totalAdjustedFitness = adjustedFitnessSums.Sum();
            var populationSize = currentGeneration.Count;
            var offspringGenomes = new List<NeatGenome>();

            for (var speciesIndex = 0; speciesIndex < speciesGroups.Count; speciesIndex++)
            {
                var group = speciesGroups[speciesIndex].OrderByDescending(individual => individual.Fitness).ToList();

                var share = totalAdjustedFitness <= 0f
                    ? populationSize / speciesGroups.Count
                    : (int)MathF.Round(populationSize * adjustedFitnessSums[speciesIndex] / totalAdjustedFitness);

                share = Math.Max(1, share);

                var eliteCount = group.Count >= _options.MinSpeciesSizeForElite
                    ? Math.Min(_options.EliteCountPerSpecies, share)
                    : 0;

                for (var i = 0; i < eliteCount; i++)
                {
                    offspringGenomes.Add(group[i].Genome.Clone());
                }

                for (var i = eliteCount; i < share; i++)
                {
                    var parentA = group[_random.Next(group.Count)];
                    NeatGenome child;

                    if (group.Count > 1 && _random.NextDouble() < _options.CrossoverRate)
                    {
                        var parentB = group[_random.Next(group.Count)];
                        child = parentA.Fitness >= parentB.Fitness
                            ? NeatGenome.Crossover(parentA.Genome, parentB.Genome, _random)
                            : NeatGenome.Crossover(parentB.Genome, parentA.Genome, _random);
                    }
                    else
                    {
                        child = parentA.Genome.Clone();
                    }

                    if (_random.NextDouble() < _options.AddConnectionRate)
                    {
                        child.TryMutateAddConnection(_random, _tracker, _options.AddConnectionMaxAttempts);
                    }

                    if (_random.NextDouble() < _options.AddNodeRate)
                    {
                        child.MutateAddNode(_random, _tracker);
                    }

                    child.MutateWeights(_random, _options.WeightPerturbRate, _options.WeightPerturbStrength, _options.WeightResetRate);

                    offspringGenomes.Add(child);
                }
            }

            while (offspringGenomes.Count < populationSize)
            {
                var fallbackSpecies = speciesGroups[_random.Next(speciesGroups.Count)];
                var parent = fallbackSpecies[_random.Next(fallbackSpecies.Count)];
                offspringGenomes.Add(parent.Genome.Clone());
            }

            if (offspringGenomes.Count > populationSize)
            {
                offspringGenomes = offspringGenomes.Take(populationSize).ToList();
            }

            return offspringGenomes.Select(genome => _agentFactory(genome)).ToList();
        }

        private List<List<(NeatGenome Genome, float Fitness)>> Speciate(List<(NeatGenome Genome, float Fitness)> population)
        {
            var speciesGroups = new List<List<(NeatGenome Genome, float Fitness)>>();
            var representatives = new List<NeatGenome>();

            foreach (var individual in population)
            {
                var placed = false;

                for (var i = 0; i < representatives.Count; i++)
                {
                    var distance = NeatGenome.CompatibilityDistance(
                        individual.Genome,
                        representatives[i],
                        _options.ExcessCoefficient,
                        _options.DisjointCoefficient,
                        _options.WeightCoefficient);

                    if (distance < _options.CompatibilityThreshold)
                    {
                        speciesGroups[i].Add(individual);
                        placed = true;
                        break;
                    }
                }

                if (!placed)
                {
                    representatives.Add(individual.Genome);
                    speciesGroups.Add(new List<(NeatGenome, float)> { individual });
                }
            }

            return speciesGroups;
        }
    }
}
