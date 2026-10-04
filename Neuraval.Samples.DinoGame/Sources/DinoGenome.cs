namespace Neuraval.Samples.DinoGame.Sources
{
    public class DinoGenome
    {
        public int EmbeddingDim { get; set; }
        public int HiddenDim { get; set; }
        public float[] Weights1 { get; set; } = System.Array.Empty<float>();
        public float[] Bias1 { get; set; } = System.Array.Empty<float>();
        public float[] Weights2 { get; set; } = System.Array.Empty<float>();
        public float[] Bias2 { get; set; } = System.Array.Empty<float>();

        public float Fitness { get; set; }
    }
}
