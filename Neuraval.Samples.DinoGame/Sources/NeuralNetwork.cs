using System;
using Neuraval.Core.Models;
using Neuraval.Evolution;

namespace Neuraval.Samples.DinoGame.Sources
{
    public class NeuralNetwork : ICloneable, IMutableAgent<NeuralNetwork>
    {
        public const int InputDim = 7;
        public const int HiddenDim = 8;
        public const int OutputDim = 2;

        public const float MutationStrength = 0.15f;

        private FeedForwardNetwork network;

        private readonly float[] lastScaledInput = new float[InputDim];
        private readonly float[] lastRawOutput = new float[OutputDim];

        public NeuralNetwork()
        {
            network = new FeedForwardNetwork(InputDim, HiddenDim, Environment.TickCount + Guid.NewGuid().GetHashCode(), OutputDim);
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

            var tensorInput = new float[1, InputDim];
            for (int i = 0; i < InputDim; i++)
            {
                float scaled = ScaleInput(i, input[i]);
                tensorInput[0, i] = scaled;
                lastScaledInput[i] = scaled;
            }

            float[,] output = network.Forward(tensorInput);

            for (int i = 0; i < OutputDim; i++)
            {
                lastRawOutput[i] = output[0, i];
            }

            return (float[])lastRawOutput.Clone();
        }

        public NetworkActivationSnapshot GetActivationSnapshot()
        {
            return new NetworkActivationSnapshot
            {
                Inputs = (float[])lastScaledInput.Clone(),
                Hidden = network.GetLastHiddenSnapshot(),
                Outputs = (float[])lastRawOutput.Clone(),
                InputToHiddenWeights = network.GetWeights1Snapshot(),
                HiddenToOutputWeights = network.GetOutputWeightsSnapshot(0, 1)
            };
        }

        private const float MaxObstacleSpawnDistance = 1350f;

        private static float ScaleInput(int index, float value)
        {
            switch (index)
            {
                case 0:
                case 1:
                    return value / MaxObstacleSpawnDistance;
                case 2:
                case 5:
                    return value / 480f;
                case 3:
                case 4:
                    return value / 150f;
                case 6:
                    return value / 12f;
                default:
                    return value;
            }
        }

        public object Clone()
        {
            var state = network.SaveState();
            return new NeuralNetwork(FeedForwardNetwork.LoadState(state));
        }

        public NeuralNetwork CloneWithMutation(Random random, float mutationRate, float mutationStrength)
        {
            var state = network.SaveState();

            MutateArray(state.Weights1, random, mutationRate, mutationStrength, WeightLimit1);
            MutateArray(state.Bias1, random, mutationRate, mutationStrength, WeightLimit1);
            MutateArray(state.Weights2, random, mutationRate, mutationStrength, WeightLimit2);
            MutateArray(state.Bias2, random, mutationRate, mutationStrength, WeightLimit2);

            return new NeuralNetwork(FeedForwardNetwork.LoadState(state));
        }

        private static float WeightLimit1 => MathF.Sqrt(6.0f / (InputDim + HiddenDim));
        private static float WeightLimit2 => MathF.Sqrt(6.0f / (HiddenDim + OutputDim));

        private static void MutateArray(float[] values, Random random, float mutationRate, float mutationStrength, float limit)
        {
            if (mutationRate <= 0f || mutationStrength <= 0f)
            {
                return;
            }

            for (int i = 0; i < values.Length; i++)
            {
                if (random.NextDouble() >= mutationRate)
                {
                    continue;
                }

                float mutated = values[i] + random.NextGaussian(0f, mutationStrength);

                if (mutated > limit)
                {
                    mutated = limit;
                }
                else if (mutated < -limit)
                {
                    mutated = -limit;
                }

                values[i] = mutated;
            }
        }

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
