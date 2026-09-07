namespace Neuraval.Evolution
{
    public interface IFitnessEvaluator<TAgent, TState, TAction> where TAgent : IAgent<TState, TAction>
    {
        float Evaluate(TAgent agent, IEnvironment<TState, TAction> environment);
    }

    public sealed class EpisodeFitnessEvaluator<TAgent, TState, TAction> : IFitnessEvaluator<TAgent, TState, TAction>
        where TAgent : IAgent<TState, TAction>
    {
        private readonly int _maxSteps;

        public EpisodeFitnessEvaluator(int maxSteps)
        {
            _maxSteps = maxSteps;
        }

        public float Evaluate(TAgent agent, IEnvironment<TState, TAction> environment)
        {
            var state = environment.Reset();
            float totalReward = 0f;

            for (int step = 0; step < _maxSteps; step++)
            {
                var action = agent.Decide(state);
                var result = environment.Step(action);
                totalReward += result.Reward;
                state = result.State;

                if (result.Done)
                {
                    break;
                }
            }

            return totalReward;
        }
    }
}
