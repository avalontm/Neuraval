namespace Neuraval.Samples.DinoGame.Sources
{
    /// <summary>
    /// Representacion ligera y serializable (JSON) del "cerebro" de un
    /// dinosaurio: unicamente los pesos/bias de su <see cref="NeuralNetwork"/>.
    /// No incluye estado de optimizador porque estos agentes no se entrenan
    /// por backpropagation, solo por seleccion + mutacion (algoritmo genetico).
    /// </summary>
    public class DinoGenome
    {
        public int EmbeddingDim { get; set; }
        public int HiddenDim { get; set; }
        public float[] Weights1 { get; set; } = System.Array.Empty<float>();
        public float[] Bias1 { get; set; } = System.Array.Empty<float>();
        public float[] Weights2 { get; set; } = System.Array.Empty<float>();
        public float[] Bias2 { get; set; } = System.Array.Empty<float>();

        /// <summary>
        /// Fitness que obtuvo este genoma cuando se guardo. Solo informativo
        /// (para depuracion / mostrar progreso), no se usa para reconstruir la red.
        /// </summary>
        public float Fitness { get; set; }
    }
}
