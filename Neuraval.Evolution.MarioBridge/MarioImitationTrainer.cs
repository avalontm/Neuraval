using System;
using System.Linq;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioImitationTrainer
    {
        private const float LearningRate = 0.001f;
        private const float Beta1 = 0.9f;
        private const float Beta2 = 0.999f;
        private const float Epsilon = 1e-8f;
        private const int BatchSize = 64;
        private const float RewardWeightStrength = 2f;
        private const int ValidationPortion = 10;
        private const int Patience = 5;
        private const int TerminalBoostFrames = 30;
        private const float CompletionBoostPeak = 2.5f;
        private const float BoostGamma = 0.9f;
        private const int DeathNearFrames = 4;
        private const float DeathNearMultiplier = 0.7f;
        private const float MaxSampleWeight = 8f;

        private readonly MarioDataset _dataset;
        private readonly int _epochs;

        public MarioImitationTrainer(MarioDataset dataset, int epochs)
        {
            _dataset = dataset;
            _epochs = epochs;
        }

        public MarioPolicyNetwork Train(Random random, MarioPolicyNetwork? seed = null)
        {
            var inputCount = _dataset.InputCount;
            var hiddenSize = MarioPolicyNetwork.DefaultHiddenSize;
            var outputCount = _dataset.OutputCount;
            var sampleCount = _dataset.Samples.Count;

            var network = MarioPolicyNetwork.Create(inputCount, outputCount, random, hiddenSize);

            if (seed != null && seed.InputCount == inputCount && seed.OutputCount == outputCount && seed.HiddenSize == hiddenSize)
            {
                Array.Copy(seed.WeightsIn, network.WeightsIn, Math.Min(seed.WeightsIn.Length, network.WeightsIn.Length));
                Array.Copy(seed.BiasHidden, network.BiasHidden, Math.Min(seed.BiasHidden.Length, network.BiasHidden.Length));
                Array.Copy(seed.WeightsOut, network.WeightsOut, Math.Min(seed.WeightsOut.Length, network.WeightsOut.Length));
                Array.Copy(seed.BiasOut, network.BiasOut, Math.Min(seed.BiasOut.Length, network.BiasOut.Length));
            }

            var weightsIn = network.WeightsIn;
            var biasHidden = network.BiasHidden;
            var weightsOut = network.WeightsOut;
            var biasOut = network.BiasOut;

            var gradWeightsIn = new float[weightsIn.Length];
            var gradBiasHidden = new float[biasHidden.Length];
            var gradWeightsOut = new float[weightsOut.Length];
            var gradBiasOut = new float[biasOut.Length];

            var momentWeightsIn = new float[weightsIn.Length];
            var momentBiasHidden = new float[biasHidden.Length];
            var momentWeightsOut = new float[weightsOut.Length];
            var momentBiasOut = new float[biasOut.Length];

            var velocityWeightsIn = new float[weightsIn.Length];
            var velocityBiasHidden = new float[biasHidden.Length];
            var velocityWeightsOut = new float[weightsOut.Length];
            var velocityBiasOut = new float[biasOut.Length];

            var order = new int[sampleCount];
            for (var i = 0; i < sampleCount; i++)
            {
                order[i] = i;
            }

            Shuffle(order, random);

            var trainCount = sampleCount - sampleCount / ValidationPortion;
            if (trainCount < 1)
            {
                trainCount = sampleCount;
            }

            var hasValidation = trainCount < sampleCount;
            var samples = _dataset.Samples;
            var sampleWeights = new float[sampleCount];

            if (sampleCount > 0)
            {
                var minReward = samples.Min(sample => sample.Reward);
                var maxReward = samples.Max(sample => sample.Reward);

                for (var i = 0; i < sampleCount; i++)
                {
                    sampleWeights[i] = MarioRewardWeighting.Compute(samples[i].Reward, minReward, maxReward, RewardWeightStrength);
                }
            }

            ApplyTerminalShaping(samples, sampleWeights);

            var hiddenValues = new float[hiddenSize];
            var outputValues = new float[outputCount];
            var targetValues = new float[outputCount];
            var outputDeltas = new float[outputCount];
            var hiddenDeltas = new float[hiddenSize];
            var timestep = 0;

            var bestWeightsIn = new float[weightsIn.Length];
            var bestBiasHidden = new float[biasHidden.Length];
            var bestWeightsOut = new float[weightsOut.Length];
            var bestBiasOut = new float[biasOut.Length];

            var bestValidationLoss = float.MaxValue;
            var bestEpoch = 0;
            var patienceCount = 0;

            for (var epoch = 0; epoch < _epochs; epoch++)
            {
                var trainOrder = new int[trainCount];
                Array.Copy(order, trainOrder, trainCount);
                Shuffle(trainOrder, random);

                var totalLoss = 0f;
                var totalWeight = 0f;

                for (var batchStart = 0; batchStart < trainCount; batchStart += BatchSize)
                {
                    var batchCount = Math.Min(BatchSize, trainCount - batchStart);

                    Array.Clear(gradWeightsIn);
                    Array.Clear(gradBiasHidden);
                    Array.Clear(gradWeightsOut);
                    Array.Clear(gradBiasOut);

                    var batchWeight = 0f;

                    for (var batchIndex = 0; batchIndex < batchCount; batchIndex++)
                    {
                        var sample = samples[trainOrder[batchStart + batchIndex]];
                        var weight = sampleWeights[trainOrder[batchStart + batchIndex]];
                        batchWeight += weight;

                        var loss = Backpropagate(
                            sample.Input,
                            sample.ActionMask,
                            weight,
                            network,
                            hiddenValues,
                            outputValues,
                            targetValues,
                            outputDeltas,
                            hiddenDeltas,
                            gradWeightsIn,
                            gradBiasHidden,
                            gradWeightsOut,
                            gradBiasOut);
                        totalLoss += loss;
                    }

                    totalWeight += batchWeight;
                    timestep++;

                    var factor = 1f / batchWeight;
                    ApplyAdamUpdate(
                        weightsIn, momentWeightsIn, velocityWeightsIn, gradWeightsIn, factor, timestep);
                    ApplyAdamUpdate(
                        biasHidden, momentBiasHidden, velocityBiasHidden, gradBiasHidden, factor, timestep);
                    ApplyAdamUpdate(
                        weightsOut, momentWeightsOut, velocityWeightsOut, gradWeightsOut, factor, timestep);
                    ApplyAdamUpdate(
                        biasOut, momentBiasOut, velocityBiasOut, gradBiasOut, factor, timestep);
                }

                var epochInfo = $"[epoch {epoch + 1}/{_epochs}] loss={(sampleCount == 0 ? 0f : totalLoss / totalWeight):F4}";

                if (hasValidation)
                {
                    var validationLoss = ComputeValidationLoss(
                        order,
                        trainCount,
                        samples,
                        sampleWeights,
                        network,
                        hiddenValues,
                        outputValues,
                        targetValues);

                    epochInfo += $" val={validationLoss:F4}";

                    if (validationLoss < bestValidationLoss)
                    {
                        bestValidationLoss = validationLoss;
                        bestEpoch = epoch + 1;
                        patienceCount = 0;

                        Array.Copy(weightsIn, bestWeightsIn, weightsIn.Length);
                        Array.Copy(biasHidden, bestBiasHidden, biasHidden.Length);
                        Array.Copy(weightsOut, bestWeightsOut, weightsOut.Length);
                        Array.Copy(biasOut, bestBiasOut, biasOut.Length);
                    }
                    else
                    {
                        patienceCount++;
                    }

                    Console.WriteLine(epochInfo);

                    if (patienceCount >= Patience)
                    {
                        Console.WriteLine($"Early stopping: sin mejora de validacion por {Patience} epochs (mejor epoch {bestEpoch}, val={bestValidationLoss:F4}).");
                        break;
                    }
                }
                else
                {
                    Console.WriteLine(epochInfo);
                }
            }

            if (hasValidation && bestEpoch > 0)
            {
                Array.Copy(bestWeightsIn, weightsIn, weightsIn.Length);
                Array.Copy(bestBiasHidden, biasHidden, biasHidden.Length);
                Array.Copy(bestWeightsOut, weightsOut, weightsOut.Length);
                Array.Copy(bestBiasOut, biasOut, biasOut.Length);

                Console.WriteLine($"Restaurando pesos del mejor epoch de validacion: {bestEpoch} (val={bestValidationLoss:F4}).");
            }

            return network;
        }

        private static float Backpropagate(
            float[] input,
            SnesButton actionMask,
            float sampleWeight,
            MarioPolicyNetwork network,
            float[] hiddenValues,
            float[] outputValues,
            float[] targetValues,
            float[] outputDeltas,
            float[] hiddenDeltas,
            float[] gradWeightsIn,
            float[] gradBiasHidden,
            float[] gradWeightsOut,
            float[] gradBiasOut)
        {
            var inputCount = network.InputCount;
            var hiddenSize = network.HiddenSize;
            var outputCount = network.OutputCount;

            FeedForward(input, network, hiddenValues, outputValues);
            MarioAgentOutput.ToAgentTargets(actionMask, targetValues);

            var loss = 0f;

            for (var o = 0; o < outputCount; o++)
            {
                var probability = Math.Clamp(outputValues[o], 1e-7f, 1f - 1e-7f);
                loss += -(targetValues[o] * MathF.Log(probability) + (1f - targetValues[o]) * MathF.Log(1f - probability));
                outputDeltas[o] = sampleWeight * (outputValues[o] - targetValues[o]);
            }

            for (var o = 0; o < outputCount; o++)
            {
                for (var h = 0; h < hiddenSize; h++)
                {
                    gradWeightsOut[o * hiddenSize + h] += outputDeltas[o] * hiddenValues[h];
                }

                gradBiasOut[o] += outputDeltas[o];
            }

            for (var h = 0; h < hiddenSize; h++)
            {
                var delta = 0f;
                for (var o = 0; o < outputCount; o++)
                {
                    delta += outputDeltas[o] * network.WeightsOut[o * hiddenSize + h];
                }

                hiddenDeltas[h] = delta * (1f - hiddenValues[h] * hiddenValues[h]);
            }

            for (var h = 0; h < hiddenSize; h++)
            {
                for (var i = 0; i < inputCount; i++)
                {
                    gradWeightsIn[h * inputCount + i] += hiddenDeltas[h] * input[i];
                }

                gradBiasHidden[h] += hiddenDeltas[h];
            }

            return sampleWeight * loss;
        }

        private static void ApplyTerminalShaping(List<MarioDatasetSample> samples, float[] weights)
        {
            for (var i = 0; i < samples.Count; i++)
            {
                if (!samples[i].Done)
                {
                    continue;
                }

                var reason = samples[i].TerminalReason;
                if (reason == (int)MarioTerminalReason.LevelComplete)
                {
                    var start = Math.Max(0, i - TerminalBoostFrames + 1);
                    var decay = TerminalBoostFrames - 1;
                    for (var j = i; j >= start; j--)
                    {
                        var boost = 1f + CompletionBoostPeak * MathF.Pow(BoostGamma, decay);
                        weights[j] *= boost;
                        decay--;
                    }
                }
                else if (reason == (int)MarioTerminalReason.Death)
                {
                    weights[i] = 0f;
                    for (var d = 1; d <= DeathNearFrames && i - d >= 0; d++)
                    {
                        weights[i - d] *= DeathNearMultiplier;
                    }
                }
            }

            for (var i = 0; i < weights.Length; i++)
            {
                weights[i] = Math.Clamp(weights[i], 0f, MaxSampleWeight);
            }
        }

        private static float ComputeValidationLoss(
            int[] order,
            int trainCount,
            List<MarioDatasetSample> samples,
            float[] sampleWeights,
            MarioPolicyNetwork network,
            float[] hiddenValues,
            float[] outputValues,
            float[] targetValues)
        {
            var totalLoss = 0f;
            var totalWeight = 0f;

            for (var i = trainCount; i < order.Length; i++)
            {
                var sample = samples[order[i]];
                FeedForward(sample.Input, network, hiddenValues, outputValues);
                MarioAgentOutput.ToAgentTargets(sample.ActionMask, targetValues);

                for (var o = 0; o < targetValues.Length; o++)
                {
                    var probability = Math.Clamp(outputValues[o], 1e-7f, 1f - 1e-7f);
                    totalLoss += sampleWeights[order[i]] *
                        -(targetValues[o] * MathF.Log(probability) + (1f - targetValues[o]) * MathF.Log(1f - probability));
                }

                totalWeight += sampleWeights[order[i]];
            }

            return totalWeight <= 0f ? 0f : totalLoss / totalWeight;
        }

        private static void FeedForward(float[] input, MarioPolicyNetwork network, float[] hiddenValues, float[] outputValues)
        {
            var inputCount = network.InputCount;
            var hiddenSize = network.HiddenSize;
            var outputCount = network.OutputCount;

            for (var h = 0; h < hiddenSize; h++)
            {
                var sum = network.BiasHidden[h];
                for (var i = 0; i < inputCount; i++)
                {
                    sum += network.WeightsIn[h * inputCount + i] * input[i];
                }

                hiddenValues[h] = MathF.Tanh(sum);
            }

            for (var o = 0; o < outputCount; o++)
            {
                var sum = network.BiasOut[o];
                for (var h = 0; h < hiddenSize; h++)
                {
                    sum += network.WeightsOut[o * hiddenSize + h] * hiddenValues[h];
                }

                outputValues[o] = Sigmoid(sum);
            }
        }

        private static void ApplyAdamUpdate(float[] parameters, float[] moment, float[] velocity, float[] gradients, float factor, int timestep)
        {
            var biasCorrection1 = 1f - MathF.Pow(Beta1, timestep);
            var biasCorrection2 = 1f - MathF.Pow(Beta2, timestep);

            for (var i = 0; i < parameters.Length; i++)
            {
                var gradient = gradients[i] * factor;

                moment[i] = Beta1 * moment[i] + (1f - Beta1) * gradient;
                velocity[i] = Beta2 * velocity[i] + (1f - Beta2) * gradient * gradient;

                var momentCorrected = moment[i] / biasCorrection1;
                var velocityCorrected = velocity[i] / biasCorrection2;

                parameters[i] -= LearningRate * momentCorrected / (MathF.Sqrt(velocityCorrected) + Epsilon);
            }
        }

        private static float Sigmoid(float value)
        {
            return 1f / (1f + MathF.Exp(-value));
        }

        private static void Shuffle(int[] indices, Random random)
        {
            for (var i = indices.Length - 1; i > 0; i--)
            {
                var j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }
    }
}