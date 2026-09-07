namespace Neuraval.Evolution.Neat
{
    public sealed class NeatConnectionGene
    {
        public int InNode { get; }
        public int OutNode { get; }
        public float Weight { get; set; }
        public bool Enabled { get; set; }
        public int Innovation { get; }

        public NeatConnectionGene(int inNode, int outNode, float weight, bool enabled, int innovation)
        {
            InNode = inNode;
            OutNode = outNode;
            Weight = weight;
            Enabled = enabled;
            Innovation = innovation;
        }

        public NeatConnectionGene Clone()
        {
            return new NeatConnectionGene(InNode, OutNode, Weight, Enabled, Innovation);
        }
    }
}
