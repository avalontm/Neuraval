namespace Neuraval.Evolution.Neat
{
    public sealed class NeatNodeGene
    {
        public int Id { get; }
        public NeatNodeType Type { get; }

        public NeatNodeGene(int id, NeatNodeType type)
        {
            Id = id;
            Type = type;
        }

        public NeatNodeGene Clone()
        {
            return new NeatNodeGene(Id, Type);
        }
    }
}
