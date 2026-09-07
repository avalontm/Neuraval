using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Evolution
{
    public interface IEvolutionStrategy<TAgent>
    {
        IReadOnlyList<TAgent> NextGeneration(IReadOnlyList<TAgent> currentGeneration, IReadOnlyList<float> fitnessScores);
    }

    public sealed class ElitistMutationStrategy<TAgent> : IEvolutionStrategy<TAgent>
        where TAgent : IMutableAgent<TAgent>
    {
        private readonly Random _random;
        private readonly int _eliteCount;
        private readonly float _mutationRate;
        private readonly float _mutationStrength;

        public ElitistMutationStrategy(Random random, int eliteCount, float mutationRate, float mutationStrength)
        {
            _random = random;
            _eliteCount = eliteCount;
            _mutationRate = mutationRate;
            _mutationStrength = mutationStrength;
        }

        public IReadOnlyList<TAgent> NextGeneration(IReadOnlyList<TAgent> currentGeneration, IReadOnlyList<float> fitnessScores)
        {
            var ranked = currentGeneration
                .Select((agent, index) => (Agent: agent, Fitness: fitnessScores[index]))
                .OrderByDescending(pair => pair.Fitness)
                .ToList();

            int eliteCount = Math.Min(_eliteCount, ranked.Count);
            var nextGeneration = new List<TAgent>(currentGeneration.Count);

            for (int i = 0; i < eliteCount; i++)
            {
                nextGeneration.Add(ranked[i].Agent.CloneWithMutation(_random, 0f, 0f));
            }

            while (nextGeneration.Count < currentGeneration.Count)
            {
                var parent = ranked[_random.Next(eliteCount)].Agent;
                nextGeneration.Add(parent.CloneWithMutation(_random, _mutationRate, _mutationStrength));
            }

            return nextGeneration;
        }
    }
}
