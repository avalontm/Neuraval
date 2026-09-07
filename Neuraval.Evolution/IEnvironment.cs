namespace Neuraval.Evolution
{
    public readonly struct EnvironmentStepResult<TState>
    {
        public TState State { get; }
        public float Reward { get; }
        public bool Done { get; }

        public EnvironmentStepResult(TState state, float reward, bool done)
        {
            State = state;
            Reward = reward;
            Done = done;
        }
    }

    public interface IEnvironment<TState, TAction>
    {
        TState Reset();
        EnvironmentStepResult<TState> Step(TAction action);
    }
}
