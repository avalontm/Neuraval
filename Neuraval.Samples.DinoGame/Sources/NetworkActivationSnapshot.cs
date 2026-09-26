namespace Neuraval.Samples.DinoGame.Sources
{
    /// <summary>
    /// Foto fija (solo datos, sin comportamiento) de la ultima decision de
    /// un <see cref="NeuralNetwork"/>: que vio, que penso y que peso usado
    /// para pensarlo. No se guarda en disco ni participa del entrenamiento;
    /// existe unicamente para que <see cref="NeuralNetworkVisualizer"/>
    /// pueda dibujar "como piensa" el Dino en el HUD sin acoplar el motor
    /// de red neuronal al codigo de dibujo (MonoGame).
    /// </summary>
    public sealed class NetworkActivationSnapshot
    {
        /// <summary>Las 7 entradas ya escaladas (mismo rango que ve la red).</summary>
        public float[] Inputs;

        /// <summary>Activaciones (post-ReLU) de las neuronas de la capa oculta.</summary>
        public float[] Hidden;

        /// <summary>Las 2 salidas usadas por el Dino: [0] saltar, [1] agacharse.</summary>
        public float[] Outputs;

        /// <summary>Pesos entrada -&gt; oculta, indexados [entrada, oculta].</summary>
        public float[,] InputToHiddenWeights;

        /// <summary>Pesos oculta -&gt; salida (solo saltar/agacharse), indexados [oculta, salida].</summary>
        public float[,] HiddenToOutputWeights;
    }
}
