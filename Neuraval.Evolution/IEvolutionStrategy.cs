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

        /// <summary>
        /// </summary>
        /// <param name="randomInjectionFraction">
        /// Fraccion (0-1) de cada nueva generacion que se rellena con
        /// individuos totalmente nuevos (via <paramref name="randomAgentFactory"/>)
        /// en vez de descendencia mutada de la elite. Sin esto, el 100% de
        /// la poblacion desciende cada generacion de los pocos elite
        /// (<paramref name="eliteCount"/>) elegidos, lo que reduce cada vez
        /// mas la diversidad genetica y puede estancar el aprendizaje en un
        /// optimo local (esto era inconsistente con
        /// <c>DinoEvolutionStore.RebuildPopulation</c>, que si reserva un
        /// ~20% de la poblacion para cerebros nuevos al recargar desde
        /// disco). Requiere pasar <paramref name="randomAgentFactory"/>; si
        /// se omite, no se inyecta nada (comportamiento previo).
        /// </param>
        /// <param name="randomAgentFactory">
        /// Crea un agente nuevo con parametros aleatorios (por ejemplo,
        /// <c>() => new NeuralNetwork()</c>). Necesario si
        /// <paramref name="randomInjectionFraction"/> es mayor que 0.
        /// </param>
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

            // Descendencia mutada de la elite, dejando reservados los
            // "randomSlots" finales para individuos nuevos.
            int bredTarget = currentGeneration.Count - randomSlots;
            while (nextGeneration.Count < bredTarget)
            {
                var parent = ranked[_random.Next(eliteCount)].Agent;
                nextGeneration.Add(parent.CloneWithMutation(_random, _mutationRate, _mutationStrength));
            }

            // Relleno con sangre nueva para mantener diversidad genetica y
            // evitar que la poblacion se estanque en un optimo local.
            while (nextGeneration.Count < currentGeneration.Count)
            {
                nextGeneration.Add(_randomAgentFactory());
            }

            return nextGeneration;
        }
    }
}
