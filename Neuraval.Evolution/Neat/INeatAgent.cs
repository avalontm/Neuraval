namespace Neuraval.Evolution.Neat
{
    public interface INeatAgent<TSelf> where TSelf : INeatAgent<TSelf>
    {
        NeatGenome Genome { get; }
        TSelf WithGenome(NeatGenome genome);
    }
}
