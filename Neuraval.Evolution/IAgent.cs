using System;

namespace Neuraval.Evolution
{
    public interface IAgent<TState, TAction>
    {
        TAction Decide(TState state);
    }

    public interface IMutableAgent<TSelf> where TSelf : IMutableAgent<TSelf>
    {
        TSelf CloneWithMutation(Random random, float mutationRate, float mutationStrength);
    }
}
