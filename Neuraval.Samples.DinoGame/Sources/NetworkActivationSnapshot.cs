namespace Neuraval.Samples.DinoGame.Sources
{
    public sealed class NetworkActivationSnapshot
    {
        public float[] Inputs;

        public float[] Hidden;

        public float[] Outputs;

        public float[,] InputToHiddenWeights;

        public float[,] HiddenToOutputWeights;
    }
}
