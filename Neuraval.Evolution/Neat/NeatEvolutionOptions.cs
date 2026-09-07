namespace Neuraval.Evolution.Neat
{
    public sealed class NeatEvolutionOptions
    {
        public float ExcessCoefficient { get; init; } = 1f;
        public float DisjointCoefficient { get; init; } = 1f;
        public float WeightCoefficient { get; init; } = 0.4f;
        public float CompatibilityThreshold { get; init; } = 3f;
        public float WeightPerturbRate { get; init; } = 0.8f;
        public float WeightPerturbStrength { get; init; } = 0.5f;
        public float WeightResetRate { get; init; } = 0.1f;
        public float AddConnectionRate { get; init; } = 0.08f;
        public float AddNodeRate { get; init; } = 0.03f;
        public float CrossoverRate { get; init; } = 0.75f;
        public int EliteCountPerSpecies { get; init; } = 2;
        public int MinSpeciesSizeForElite { get; init; } = 3;
        public int AddConnectionMaxAttempts { get; init; } = 20;
    }
}
