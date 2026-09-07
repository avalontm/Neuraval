using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Evolution
{
    public sealed class Population<TAgent, TState, TAction> where TAgent : IAgent<TState, TAction>
    {
        private readonly Func<IEnvironment<TState, TAction>> _environmentFactory;
        private readonly IFitnessEvaluator<TAgent, TState, TAction> _fitnessEvaluator;
        private readonly IEvolutionStrategy<TAgent> _evolutionStrategy;

        public IReadOnlyList<TAgent> Agents { get; private set; }
        public IReadOnlyList<float> LastFitnessScores { get; private set; }
        public int Generation { get; private set; }

        public float BestFitness => LastFitnessScores.Count == 0 ? 0f : LastFitnessScores.Max();
        public float AverageFitness => LastFitnessScores.Count == 0 ? 0f : LastFitnessScores.Average();

        public Population(
            IReadOnlyList<TAgent> initialAgents,
            Func<IEnvironment<TState, TAction>> environmentFactory,
            IFitnessEvaluator<TAgent, TState, TAction> fitnessEvaluator,
            IEvolutionStrategy<TAgent> evolutionStrategy)
        {
            Agents = initialAgents;
            _environmentFactory = environmentFactory;
            _fitnessEvaluator = fitnessEvaluator;
            _evolutionStrategy = evolutionStrategy;
            LastFitnessScores = Array.Empty<float>();
            Generation = 0;
        }

        public IReadOnlyList<float> EvaluateGeneration()
        {
            var scores = new float[Agents.Count];

            for (int i = 0; i < Agents.Count; i++)
            {
                var environment = _environmentFactory();
                scores[i] = _fitnessEvaluator.Evaluate(Agents[i], environment);
            }

            LastFitnessScores = scores;
            return scores;
        }

        public void Advance()
        {
            Agents = _evolutionStrategy.NextGeneration(Agents, LastFitnessScores);
            Generation++;
        }
    }
}
