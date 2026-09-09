namespace Neuraval.Evolution.Neat
{
    public sealed class NeatEvolutionOptions
    {
        public float ExcessCoefficient { get; set; } = 1f;
        public float DisjointCoefficient { get; set; } = 1f;
        public float WeightCoefficient { get; set; } = 0.4f;
        public float CompatibilityThreshold { get; set; } = 3f;
        public float WeightPerturbRate { get; set; } = 0.8f;
        public float WeightPerturbStrength { get; set; } = 0.5f;
        public float WeightResetRate { get; set; } = 0.1f;
        public float AddConnectionRate { get; set; } = 0.08f;
        public float AddNodeRate { get; set; } = 0.03f;
        public float CrossoverRate { get; set; } = 0.75f;
        public int EliteCountPerSpecies { get; set; } = 2;
        public int MinSpeciesSizeForElite { get; set; } = 3;
        public int AddConnectionMaxAttempts { get; set; } = 20;
    }
}
