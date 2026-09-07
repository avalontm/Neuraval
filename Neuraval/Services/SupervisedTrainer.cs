using System;
using System.Linq;
using System.Collections.Generic;
using Neuraval.Abstractions;
using Neuraval.Core.Models;
using Neuraval.Core.Utils;

namespace Neuraval.Core.Services
{
    public class SupervisedTrainer : ITrainer<TransformerModel, CausalTrainingDataset>
    {
        private readonly TransformerModel _model;
        private readonly float _learningRate;
        private float _currentLearningRate;
        private List<float> _trainingLosses;
        private List<float> _validationLosses;

        private readonly Random _shuffleRandom;
        private readonly int _padToken;

        private int[,]? _tokenBatchBuffer;
        private int[]? _validLengthsBuffer;
        private int[]? _lossStartIndicesBuffer;

        private float _bestValidationLoss = float.MaxValue;
        private int _epochsWithoutImprovement;
        private int _lastCompletedEpoch = -1;

        public float BestValidationLoss => _bestValidationLoss;
        public int EpochsWithoutImprovement => _epochsWithoutImprovement;
        public int LastCompletedEpoch => _lastCompletedEpoch;

        public SupervisedTrainer(TransformerModel model, float learningRate = 0.001f, int shuffleSeed = 42, int padToken = 0)
        {
            _model = model;
            _learningRate = learningRate;
            _currentLearningRate = learningRate;
            _trainingLosses = new List<float>();
            _validationLosses = new List<float>();
            _shuffleRandom = new Random(shuffleSeed);
            _padToken = padToken;
        }

        public void Train(TransformerModel model, CausalTrainingDataset dataset)
        {
            if (!ReferenceEquals(model, _model))
            {
                throw new InvalidOperationException("This trainer instance is bound to a different model.");
            }

            TrainCausalWithValidation(
                dataset.TrainingExamples,
                dataset.ValidationExamples,
                dataset.Epochs,
                dataset.BatchSize,
                dataset.Patience,
                dataset.OnEpochCompleted,
                dataset.ResumeFrom,
                dataset.CheckpointIntervalEpochs,
                dataset.OnCheckpoint,
                dataset.LogFilePath,
                dataset.GradientAccumulationSteps);
        }

        public void Train(
            List<int[]> inputSequences,
            List<int[]> targetSequences,
            int epochs,
            int batchSize = 32,
            Action<int, float, float>? onEpochCompleted = null)
        {
            if (inputSequences.Count != targetSequences.Count)
            {
                throw new ArgumentException("Input and target sequences must have same count");
            }

            _trainingLosses.Clear();

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float epochLoss = 0;
                int numBatches = 0;

                var indices = Enumerable.Range(0, inputSequences.Count).OrderBy(_ => _shuffleRandom.Next()).ToList();

                for (int i = 0; i < inputSequences.Count; i += batchSize)
                {
                    int currentBatchSize = Math.Min(batchSize, inputSequences.Count - i);
                    float batchLoss = 0;

                    _model.ZeroGradients();

                    for (int j = 0; j < currentBatchSize; j++)
                    {
                        int idx = indices[i + j];
                        var input = inputSequences[idx];
                        var target = targetSequences[idx];

                        float loss = _model.CalculateLoss(input, target);
                        batchLoss += loss;
                    }

                    batchLoss /= currentBatchSize;
                    epochLoss += batchLoss;
                    numBatches++;

                    _model.AverageGradients(currentBatchSize);
                    _model.ClipGradients(1.0f);
                    _model.UpdateWeights(_currentLearningRate);
                }

                epochLoss /= numBatches;
                _trainingLosses.Add(epochLoss);

                _currentLearningRate = GetLearningRateWithWarmup(epoch);

                onEpochCompleted?.Invoke(epoch, epochLoss, 0.0f);

                if (onEpochCompleted == null && epoch % 10 == 0)
                {
                    System.Console.WriteLine($"Epoch {epoch}/{epochs} - Loss: {epochLoss:F6} - LR: {_currentLearningRate:F6}");
                }
            }
        }

        public void TrainWithValidation(
    List<int[]> trainingInputs,
    List<int[]> trainingTargets,
    List<int[]> validationInputs,
    List<int[]> validationTargets,
    int epochs,
    int batchSize = 32,
    int patience = 20,
    Action<int, float, float>? onEpochCompleted = null)
        {
            if (trainingInputs.Count != trainingTargets.Count)
            {
                throw new ArgumentException("Training inputs and targets must have same count");
            }

            if (validationInputs.Count != validationTargets.Count)
            {
                throw new ArgumentException("Validation inputs and targets must have same count");
            }

            _trainingLosses.Clear();
            _validationLosses.Clear();

            float bestValidationLoss = float.MaxValue;
            int epochsWithoutImprovement = 0;
            TransformerModelState? bestModelState = null;
            float minDelta = 1e-5f;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var (trainingLoss, _, skippedBatches) = TrainEpoch(trainingInputs, trainingTargets, batchSize, epoch);

                if (float.IsNaN(trainingLoss))
                {
                    Console.WriteLine($"\nEpoch {epoch} produced no valid batches (all skipped due to NaN/Infinity). Skipping epoch.");
                    continue;
                }

                if (skippedBatches > 0)
                {
                    Console.WriteLine($"Epoch {epoch}: {skippedBatches} batch(es) skipped due to NaN/Infinity.");
                }

                _trainingLosses.Add(trainingLoss);

                float validationLoss = ValidateEpoch(validationInputs, validationTargets);
                _validationLosses.Add(validationLoss);

                _lastCompletedEpoch = epoch;

                onEpochCompleted?.Invoke(epoch, trainingLoss, validationLoss);

                if (validationLoss < bestValidationLoss - minDelta)
                {
                    bestValidationLoss = validationLoss;
                    bestModelState = _model.SaveState();
                    epochsWithoutImprovement = 0;
                }
                else
                {
                    epochsWithoutImprovement++;
                }

                _bestValidationLoss = bestValidationLoss;
                _epochsWithoutImprovement = epochsWithoutImprovement;

                if (epochsWithoutImprovement >= patience)
                {
                    Console.WriteLine($"\nEarly stopping at epoch {epoch}. Best validation loss: {bestValidationLoss:F6}");
                    break;
                }
            }

            if (bestModelState != null)
            {
                _model.RestoreFrom(bestModelState);
                Console.WriteLine($"Restored best model weights (validation loss: {bestValidationLoss:F6}).");
            }
        }

        public void TrainCausalWithValidation(
            List<CausalExample> trainingExamples,
            List<CausalExample> validationExamples,
            int epochs,
            int batchSize = 32,
            int patience = 20,
            Action<int, float, float>? onEpochCompleted = null,
            TrainingProgressState? resumeFrom = null,
            int checkpointIntervalEpochs = 0,
            Action<TrainingProgressState>? onCheckpoint = null,
            string? logFilePath = null,
            int gradientAccumulationSteps = 1)
        {
            int startEpoch = 0;
            float bestValidationLoss = float.MaxValue;
            int epochsWithoutImprovement = 0;

            if (resumeFrom != null)
            {
                _trainingLosses = new List<float>(resumeFrom.TrainingLosses);
                _validationLosses = new List<float>(resumeFrom.ValidationLosses);
                bestValidationLoss = resumeFrom.BestValidationLoss;
                epochsWithoutImprovement = resumeFrom.EpochsWithoutImprovement;
                startEpoch = resumeFrom.LastCompletedEpoch + 1;
            }
            else
            {
                _trainingLosses.Clear();
                _validationLosses.Clear();
            }

            TransformerModelState? bestModelState = null;
            float minDelta = 1e-5f;

            for (int epoch = startEpoch; epoch < epochs; epoch++)
            {
                var (trainingLoss, gradientNorm, skippedBatches) = TrainCausalEpoch(trainingExamples, batchSize, epoch, gradientAccumulationSteps);

                if (float.IsNaN(trainingLoss))
                {
                    Console.WriteLine($"\nEpoch {epoch} produced no valid batches (all skipped due to NaN/Infinity). Skipping epoch.");
                    continue;
                }

                if (skippedBatches > 0)
                {
                    Console.WriteLine($"Epoch {epoch}: {skippedBatches} batch(es) skipped due to NaN/Infinity.");
                }

                _trainingLosses.Add(trainingLoss);

                float validationLoss = ValidateCausalEpoch(validationExamples, batchSize);
                _validationLosses.Add(validationLoss);

                _lastCompletedEpoch = epoch;

                onEpochCompleted?.Invoke(epoch, trainingLoss, validationLoss);

                if (logFilePath != null)
                {
                    TrainingLogger.LogEpoch(logFilePath, epoch, trainingLoss, validationLoss, _currentLearningRate, gradientNorm);
                }

                if (validationLoss < bestValidationLoss - minDelta)
                {
                    bestValidationLoss = validationLoss;
                    bestModelState = _model.SaveState();
                    epochsWithoutImprovement = 0;
                }
                else
                {
                    epochsWithoutImprovement++;
                }

                _bestValidationLoss = bestValidationLoss;
                _epochsWithoutImprovement = epochsWithoutImprovement;

                if (checkpointIntervalEpochs > 0 && onCheckpoint != null && (epoch + 1) % checkpointIntervalEpochs == 0)
                {
                    onCheckpoint(GetTrainingProgress(epoch, bestValidationLoss, epochsWithoutImprovement));
                }

                if (epochsWithoutImprovement >= patience)
                {
                    Console.WriteLine($"\nEarly stopping at epoch {epoch}. Best validation loss: {bestValidationLoss:F6}");
                    break;
                }
            }

            if (bestModelState != null)
            {
                _model.RestoreFrom(bestModelState);
                Console.WriteLine($"Restored best model weights (validation loss: {bestValidationLoss:F6}).");
            }
        }

        public TrainingProgressState GetTrainingProgress(int lastCompletedEpoch, float bestValidationLoss, int epochsWithoutImprovement)
        {
            return new TrainingProgressState
            {
                LastCompletedEpoch = lastCompletedEpoch,
                TrainingLosses = new List<float>(_trainingLosses),
                ValidationLosses = new List<float>(_validationLosses),
                BestValidationLoss = bestValidationLoss,
                EpochsWithoutImprovement = epochsWithoutImprovement
            };
        }

        private (int[,] tokenBatch, int[] validLengths, int[] lossStartIndices) BuildPaddedCausalBatch(
            List<CausalExample> examples, List<int> indices, int start, int end)
        {
            int batchSize = end - start;
            int maxLen = 0;

            for (int j = start; j < end; j++)
            {
                maxLen = Math.Max(maxLen, examples[indices[j]].Tokens.Length);
            }

            var tokenBatch = (_tokenBatchBuffer != null &&
                               _tokenBatchBuffer.GetLength(0) == batchSize &&
                               _tokenBatchBuffer.GetLength(1) == maxLen)
                ? _tokenBatchBuffer
                : new int[batchSize, maxLen];
            _tokenBatchBuffer = tokenBatch;

            var validLengths = (_validLengthsBuffer != null && _validLengthsBuffer.Length == batchSize)
                ? _validLengthsBuffer
                : new int[batchSize];
            _validLengthsBuffer = validLengths;

            var lossStartIndices = (_lossStartIndicesBuffer != null && _lossStartIndicesBuffer.Length == batchSize)
                ? _lossStartIndicesBuffer
                : new int[batchSize];
            _lossStartIndicesBuffer = lossStartIndices;

            for (int j = start; j < end; j++)
            {
                int b = j - start;
                var example = examples[indices[j]];
                var tokens = example.Tokens;

                validLengths[b] = tokens.Length;
                lossStartIndices[b] = example.ResponseStartIndex;

                for (int i = 0; i < maxLen; i++)
                {
                    tokenBatch[b, i] = i < tokens.Length ? tokens[i] : _padToken;
                }
            }

            return (tokenBatch, validLengths, lossStartIndices);
        }

        private (float loss, float gradientNorm, int skippedBatches) TrainCausalEpoch(List<CausalExample> examples, int batchSize, int epoch, int gradientAccumulationSteps = 1)
        {
            if (gradientAccumulationSteps < 1)
            {
                gradientAccumulationSteps = 1;
            }

            var indices = Enumerable.Range(0, examples.Count).OrderBy(_ => _shuffleRandom.Next()).ToList();

            float totalLoss = 0;
            float totalGradientNorm = 0;
            int numUpdates = 0;
            int skippedBatches = 0;

            int microBatchesInCycle = 0;
            float accumulatedLossSum = 0;
            int accumulatedPredictedPositions = 0;
            bool cycleHasNaN = false;
            int cycleStartIndex = -1;

            for (int i = 0; i < examples.Count; i += batchSize)
            {
                int end = Math.Min(i + batchSize, examples.Count);

                if (microBatchesInCycle == 0)
                {
                    _model.ZeroGradients();
                    accumulatedLossSum = 0;
                    accumulatedPredictedPositions = 0;
                    cycleHasNaN = false;
                    cycleStartIndex = i;
                }

                var (tokenBatch, validLengths, lossStartIndices) = BuildPaddedCausalBatch(examples, indices, i, end);
                float batchLoss = _model.CalculateCausalLossBatch(tokenBatch, validLengths, lossStartIndices);
                int predictedPositions = _model.LastPredictedPositions;

                if (predictedPositions > 0 && !IsInvalidNumber(batchLoss))
                {
                    accumulatedLossSum += batchLoss * predictedPositions;
                    accumulatedPredictedPositions += predictedPositions;
                }
                else
                {
                    cycleHasNaN = true;
                }

                microBatchesInCycle++;
                bool lastMicroBatchOverall = end >= examples.Count;

                if (microBatchesInCycle < gradientAccumulationSteps && !lastMicroBatchOverall)
                {
                    continue;
                }

                microBatchesInCycle = 0;

                if (accumulatedPredictedPositions == 0)
                {
                    skippedBatches++;
                    continue;
                }

                _model.AverageGradients(accumulatedPredictedPositions);
                float gradientNorm = _model.ClipGradients(1.0f);

                if (cycleHasNaN || IsInvalidNumber(gradientNorm))
                {
                    skippedBatches++;
                    Console.WriteLine($"\nWarning: NaN/Infinity detected at epoch {epoch}, batch starting at index {cycleStartIndex}. Skipping weight update for this batch.");
                    continue;
                }

                _currentLearningRate = GetLearningRateWithWarmup(epoch);
                _model.UpdateWeights(_currentLearningRate);

                totalLoss += accumulatedLossSum / accumulatedPredictedPositions;
                totalGradientNorm += gradientNorm;
                numUpdates++;
            }

            if (numUpdates == 0)
            {
                return (float.NaN, float.NaN, skippedBatches);
            }

            return (totalLoss / numUpdates, totalGradientNorm / numUpdates, skippedBatches);
        }

        private float ValidateCausalEpoch(List<CausalExample> examples, int batchSize)
        {
            if (examples.Count == 0)
            {
                return 0;
            }

            var indices = Enumerable.Range(0, examples.Count).ToList();

            float totalLoss = 0;
            int numBatches = 0;

            for (int i = 0; i < examples.Count; i += batchSize)
            {
                int end = Math.Min(i + batchSize, examples.Count);

                var (tokenBatch, validLengths, lossStartIndices) = BuildPaddedCausalBatch(examples, indices, i, end);
                totalLoss += _model.CalculateCausalLossBatch(tokenBatch, validLengths, lossStartIndices);
                numBatches++;
            }

            return totalLoss / numBatches;
        }

        private float GetLearningRateWithWarmup(int epoch, int warmupEpochs = 10)
        {
            if (epoch < warmupEpochs)
            {
                return _learningRate * (epoch + 1.0f) / warmupEpochs;
            }
            else
            {
                return _learningRate * MathF.Pow(0.95f, epoch - warmupEpochs);
            }
        }

        private static bool IsInvalidNumber(float value)
        {
            return float.IsNaN(value) || float.IsInfinity(value);
        }

        private (float loss, float gradientNorm, int skippedBatches) TrainEpoch(List<int[]> inputs, List<int[]> targets, int batchSize, int epoch)
        {
            if (inputs.Count != targets.Count)
            {
                throw new ArgumentException($"Input and target count mismatch: inputs={inputs.Count}, targets={targets.Count}");
            }

            var indices = Enumerable.Range(0, inputs.Count).OrderBy(_ => _shuffleRandom.Next()).ToList();

            float totalLoss = 0;
            float totalGradientNorm = 0;
            int numBatches = 0;
            int skippedBatches = 0;
            int totalBatches = (inputs.Count + batchSize - 1) / batchSize;

            for (int i = 0; i < inputs.Count; i += batchSize)
            {
                int end = Math.Min(i + batchSize, inputs.Count);
                float batchLoss = 0;
                int currentBatchSize = end - i;

                _model.ZeroGradients();

                for (int j = i; j < end; j++)
                {
                    int idx = indices[j];
                    var input = inputs[idx];
                    var target = targets[idx];

                    try
                    {
                        float loss = _model.CalculateLoss(input, target);
                        batchLoss += loss;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"\nError processing training sample {idx}:");
                        Console.WriteLine($"  Input length: {input?.Length ?? -1}");
                        Console.WriteLine($"  Target length: {target?.Length ?? -1}");
                        Console.WriteLine($"  Exception: {ex.Message}");
                        throw;
                    }
                }

                batchLoss /= currentBatchSize;

                _model.AverageGradients(currentBatchSize);
                float gradientNorm = _model.ClipGradients(1.0f);

                if (IsInvalidNumber(batchLoss) || IsInvalidNumber(gradientNorm))
                {
                    skippedBatches++;
                    Console.WriteLine($"\nWarning: NaN/Infinity detected at epoch {epoch}, batch starting at index {i}. Skipping weight update for this batch.");
                    continue;
                }

                _currentLearningRate = GetLearningRateWithWarmup(epoch);
                _model.UpdateWeights(_currentLearningRate);

                totalLoss += batchLoss;
                totalGradientNorm += gradientNorm;
                numBatches++;

                if (numBatches % 10 == 0)
                {
                    float progress = (float)numBatches / totalBatches * 100;
                    Console.Write($"\r  Batches: {numBatches}/{totalBatches} ({progress:F1}%) - LR: {_currentLearningRate:F6}   ");
                }
            }

            Console.WriteLine();

            if (numBatches == 0)
            {
                return (float.NaN, float.NaN, skippedBatches);
            }

            return (totalLoss / numBatches, totalGradientNorm / numBatches, skippedBatches);
        }

        private float ValidateEpoch(List<int[]> inputs, List<int[]> targets)
        {
            float totalLoss = 0;

            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var target = targets[i];

                try
                {
                    float loss = _model.CalculateLoss(input, target);
                    totalLoss += loss;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\nError validating sample {i}:");
                    Console.WriteLine($"  Input length: {input?.Length ?? -1}");
                    Console.WriteLine($"  Target length: {target?.Length ?? -1}");
                    Console.WriteLine($"  Exception: {ex.Message}");
                    throw;
                }

                if ((i + 1) % 50 == 0 || i == inputs.Count - 1)
                {
                    float progress = (float)(i + 1) / inputs.Count * 100;
                    Console.Write($"\r  Validating: {i + 1}/{inputs.Count} ({progress:F1}%)   ");
                }
            }

            Console.WriteLine();
            return totalLoss / inputs.Count;
        }

        public float Evaluate(List<int[]> inputs, List<int[]> targets)
        {
            if (inputs.Count != targets.Count)
            {
                throw new ArgumentException("Inputs and targets must have same count");
            }

            var losses = new List<float>();
            int correctPredictions = 0;
            int totalPredictions = 0;

            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var target = targets[i];

                float loss = _model.CalculateLoss(input, target);
                losses.Add(loss);

                for (int j = 0; j < Math.Min(input.Length, target.Length); j++)
                {
                    int predictedToken = _model.PredictNextToken(input.Take(j + 1).ToArray());

                    if (j < target.Length && predictedToken == target[j])
                    {
                        correctPredictions++;
                    }
                    totalPredictions++;
                }
            }

            float avgLoss = losses.Sum() / losses.Count;
            float accuracy = (float)correctPredictions / totalPredictions;

            System.Console.WriteLine($"Evaluation - Loss: {avgLoss:F6}, Accuracy: {accuracy:P2}");

            return avgLoss;
        }

        public List<float> GetTrainingLosses()
        {
            return new List<float>(_trainingLosses);
        }

        public List<float> GetValidationLosses()
        {
            return new List<float>(_validationLosses);
        }

        public void SaveTrainingHistory(string filepath)
        {
            var history = new TrainingHistory
            {
                TrainingLosses = _trainingLosses,
                ValidationLosses = _validationLosses
            };

            var json = System.Text.Json.JsonSerializer.Serialize(history, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });

            System.IO.File.WriteAllText(filepath, json);
        }

        public void LoadTrainingHistory(string filepath)
        {
            if (!System.IO.File.Exists(filepath))
            {
                throw new System.IO.FileNotFoundException($"Training history file not found: {filepath}");
            }

            var json = System.IO.File.ReadAllText(filepath);
            var history = System.Text.Json.JsonSerializer.Deserialize<TrainingHistory>(json);

            if (history != null)
            {
                _trainingLosses = history.TrainingLosses ?? new List<float>();
                _validationLosses = history.ValidationLosses ?? new List<float>();
            }
        }
    }

    public class TrainingHistory
    {
        public List<float>? TrainingLosses { get; set; }
        public List<float>? ValidationLosses { get; set; }
    }

    public class TrainingProgressState
    {
        public int LastCompletedEpoch { get; set; }
        public List<float> TrainingLosses { get; set; } = new List<float>();
        public List<float> ValidationLosses { get; set; } = new List<float>();
        public float BestValidationLoss { get; set; }
        public int EpochsWithoutImprovement { get; set; }
    }

    public class CausalTrainingDataset
    {
        public List<CausalExample> TrainingExamples { get; set; } = new List<CausalExample>();
        public List<CausalExample> ValidationExamples { get; set; } = new List<CausalExample>();
        public int Epochs { get; set; }
        public int BatchSize { get; set; } = 32;
        public int Patience { get; set; } = 20;
        public int GradientAccumulationSteps { get; set; } = 1;
        public int CheckpointIntervalEpochs { get; set; }
        public string? LogFilePath { get; set; }
        public TrainingProgressState? ResumeFrom { get; set; }
        public Action<int, float, float>? OnEpochCompleted { get; set; }
        public Action<TrainingProgressState>? OnCheckpoint { get; set; }
    }
}