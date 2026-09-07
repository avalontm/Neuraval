using System;
using Neuraval.Core.Models;
using Neuraval.Evolution;

namespace Neuraval.Samples.DinoGame.Sources
{
    /// <summary>
    /// Red neuronal del Dino.
    ///
    /// Ya no depende de Accord.Neuro/Accord.Statistics: usa
    /// <see cref="FeedForwardNetwork"/>, el motor de red neuronal propio del
    /// proyecto (sin librerias externas, en Neuraval.Core.Models).
    ///
    /// Ademas implementa <see cref="IMutableAgent{TSelf}"/> para poder
    /// reproducirse dentro de un algoritmo genetico (elitismo + mutacion) y
    /// expone su genoma (pesos) para poder guardarlo/cargarlo en disco y
    /// asi conservar el aprendizaje entre partidas.
    /// </summary>
    public class NeuralNetwork : ICloneable, IMutableAgent<NeuralNetwork>
    {
        // FeedForwardNetwork es simetrica (entrada y salida comparten
        // dimension), asi que se le pasan las 7 caracteristicas del juego
        // como entrada/salida y solo se leen las 2 primeras posiciones de la
        // salida (salto, agacharse) descartando el resto.
        public const int InputDim = 7;
        public const int HiddenDim = 8;

        private FeedForwardNetwork network;

        public NeuralNetwork()
        {
            // Sin entrenamiento: cada dinosaurio nace con pesos aleatorios,
            // igual que la version con Accord (NguyenWidrow.Randomize()).
            network = new FeedForwardNetwork(InputDim, HiddenDim, Environment.TickCount + Guid.NewGuid().GetHashCode());
        }

        private NeuralNetwork(FeedForwardNetwork network)
        {
            this.network = network;
        }

        public float[] Predict(float[] input)
        {
            if (input.Length != InputDim)
            {
                throw new ArgumentException($"Se esperaban {InputDim} entradas, se recibieron {input.Length}");
            }

            // FeedForwardNetwork trabaja sobre una secuencia [pasos, dimension];
            // aqui solo hace falta un unico paso (una decision por fotograma).
            var tensorInput = new float[1, InputDim];
            for (int i = 0; i < InputDim; i++)
            {
                tensorInput[0, i] = ScaleInput(i, input[i]);
            }

            float[,] output = network.Forward(tensorInput);

            // Solo se usan las 2 primeras salidas como decision:
            // [0] = saltar, [1] = agacharse. El resto del vector se descarta.
            return new float[] { output[0, 0], output[0, 1] };
        }

        // La red se inicializa con pesos pequeños (Xavier/Glorot), asi que
        // conviene llevar las magnitudes del juego (pixeles, velocidad) a un
        // rango razonable antes de la primera capa, en vez de pasarlas en
        // crudo como hacia la version anterior.
        private static float ScaleInput(int index, float value)
        {
            switch (index)
            {
                case 0: // distancia al obstaculo (pixeles)
                case 1: // posicion X del obstaculo (pixeles)
                    return value / 500f;
                case 2: // posicion Y del obstaculo (pixeles)
                case 5: // posicion Y del dino (pixeles)
                    return value / 480f;
                case 3: // ancho del obstaculo (pixeles)
                case 4: // alto del obstaculo (pixeles)
                    return value / 150f;
                case 6: // velocidad del juego
                    return value / 12f;
                default:
                    return value;
            }
        }

        public object Clone()
        {
            // Clonado real por valor (copia el estado de los pesos), a
            // diferencia del MemberwiseClone superficial que usaba la
            // version con Accord.Neuro.
            var state = network.SaveState();
            return new NeuralNetwork(FeedForwardNetwork.LoadState(state));
        }

        /// <summary>
        /// Crea una copia de esta red y aplica mutacion gaussiana a sus
        /// pesos/bias con probabilidad <paramref name="mutationRate"/> por
        /// parametro. Es la operacion de "reproduccion asexual" usada por el
        /// algoritmo genetico para generar la siguiente generacion a partir
        /// de los dinosaurios con mejor fitness.
        /// Con mutationRate = 0 y mutationStrength = 0 equivale a un clon
        /// exacto (se usa para preservar a los "elite" sin alterarlos).
        /// </summary>
        public NeuralNetwork CloneWithMutation(Random random, float mutationRate, float mutationStrength)
        {
            var state = network.SaveState();

            MutateArray(state.Weights1, random, mutationRate, mutationStrength);
            MutateArray(state.Bias1, random, mutationRate, mutationStrength);
            MutateArray(state.Weights2, random, mutationRate, mutationStrength);
            MutateArray(state.Bias2, random, mutationRate, mutationStrength);

            return new NeuralNetwork(FeedForwardNetwork.LoadState(state));
        }

        private static void MutateArray(float[] values, Random random, float mutationRate, float mutationStrength)
        {
            if (mutationRate <= 0f || mutationStrength <= 0f)
            {
                return;
            }

            for (int i = 0; i < values.Length; i++)
            {
                if (random.NextDouble() < mutationRate)
                {
                    values[i] += random.NextGaussian(0f, mutationStrength);
                }
            }
        }

        /// <summary>
        /// Exporta el genoma (pesos y bias) como una estructura ligera y
        /// serializable en JSON, sin el estado del optimizador Adam (no hace
        /// falta: estos dinosaurios nunca se entrenan por backpropagation,
        /// solo evolucionan por seleccion + mutacion).
        /// </summary>
        public DinoGenome ExportGenome()
        {
            var state = network.SaveState();
            return new DinoGenome
            {
                EmbeddingDim = state.EmbeddingDim,
                HiddenDim = state.HiddenDim,
                Weights1 = state.Weights1,
                Bias1 = state.Bias1,
                Weights2 = state.Weights2,
                Bias2 = state.Bias2
            };
        }

        /// <summary>
        /// Reconstruye una red neuronal a partir de un genoma previamente
        /// exportado (por ejemplo, cargado desde disco).
        /// </summary>
        public static NeuralNetwork FromGenome(DinoGenome genome)
        {
            var state = new FeedForwardNetworkState
            {
                EmbeddingDim = genome.EmbeddingDim,
                HiddenDim = genome.HiddenDim,
                Weights1 = genome.Weights1,
                Bias1 = genome.Bias1,
                Weights2 = genome.Weights2,
                Bias2 = genome.Bias2
            };

            return new NeuralNetwork(FeedForwardNetwork.LoadState(state));
        }
    }
}
