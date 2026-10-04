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
        private readonly float _randomInjectionFraction;
        private readonly Func<TAgent> _randomAgentFactory;

        public ElitistMutationStrategy(Random random, int eliteCount, float mutationRate, float mutationStrength)
            : this(random, eliteCount, mutationRate, mutationStrength, 0f, null)
        {
        }

        public ElitistMutationStrategy(
            Random random,
            int eliteCount,
            float mutationRate,
            float mutationStrength,
            float randomInjectionFraction,
            Func<TAgent> randomAgentFactory)
        {
            _random = random;
            _eliteCount = eliteCount;
            _mutationRate = mutationRate;
            _mutationStrength = mutationStrength;
            _randomInjectionFraction = randomInjectionFraction;
            _randomAgentFactory = randomAgentFactory;
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

            int randomSlots = 0;
            if (_randomAgentFactory != null && _randomInjectionFraction > 0f)
            {
                randomSlots = (int)(currentGeneration.Count * _randomInjectionFraction);
            }

            int bredTarget = currentGeneration.Count - randomSlots;
            while (nextGeneration.Count < bredTarget)
            {
                var parent = ranked[_random.Next(eliteCount)].Agent;
                nextGeneration.Add(parent.CloneWithMutation(_random, _mutationRate, _mutationStrength));
            }

            while (nextGeneration.Count < currentGeneration.Count)
            {
                nextGeneration.Add(_randomAgentFactory());
            }

            return nextGeneration;
        }
    }
}
