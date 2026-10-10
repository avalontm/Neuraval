using System;
using System.Collections.Generic;
using Neuraval.Core.Utils;
using Neuraval.Cuda;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class ModernDecoderModel
    {
        private readonly TransformerConfig _config;

        private EmbeddingLayer _embedding;
        private List<ModernDecoderBlock> _blocks;
        private RMSNorm _finalNorm;

        private float[,]? _outputWeights;
        private QuantizedMatrixQ8? _outputWeightsQ8;
        private float[,]? _outputWeightsGradients;
        private float[,]? _accumulatedOutputWeightsGradients;
        private AdamMatrixOptimizer? _outputWeightsOptimizer;
        private CudaWeightCache _outputProjectionCache = null!;
        private readonly bool _inferenceOnly;

        private float[,]? _pendingHiddenStateGradients;

        public int VocabSize => _config.VocabSize;
        public int HiddenSize => _config.HiddenSize;
        public int NumHiddenLayers => _config.NumHiddenLayers;
        public bool TiesWordEmbeddings => _config.TieWordEmbeddings;
        public int MaxPositionEmbeddings => _config.MaxPositionEmbeddings;

        public ModernDecoderModel(TransformerConfig config, int seed = 42)
            : this(config, seed, inferenceOnly: false)
        {
        }

        private ModernDecoderModel(TransformerConfig config, int seed, bool inferenceOnly)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            config.Validate();
            _config = config.Clone();
            _inferenceOnly = inferenceOnly;

            if (inferenceOnly)
            {
                _embedding = null!;
                _blocks = new List<ModernDecoderBlock>();
                _finalNorm = null!;
                _outputProjectionCache = new CudaWeightCache(_config.VocabSize, _config.HiddenSize);
                return;
            }

            _embedding = new EmbeddingLayer(_config.VocabSize, _config.HiddenSize, seed);

            _blocks = new List<ModernDecoderBlock>();
            for (int i = 0; i < _config.NumHiddenLayers; i++)
            {
                _blocks.Add(new ModernDecoderBlock(_config, seed + i + 1));
            }

            _finalNorm = new RMSNorm(_config.HiddenSize, _config.RmsNormEps);

            InitializeOutputProjection(seed);
        }

        private void InitializeOutputProjection(int seed)
        {
            _outputProjectionCache = new CudaWeightCache(_config.VocabSize, _config.HiddenSize);

            if (_config.TieWordEmbeddings)
                return;

            var random = new Random(seed + 777);
            float limit = MathF.Sqrt(6f / (_config.VocabSize + _config.HiddenSize));

            _outputWeights = new float[_config.VocabSize, _config.HiddenSize];
            for (int i = 0; i < _config.VocabSize; i++)
                for (int j = 0; j < _config.HiddenSize; j++)
                    _outputWeights[i, j] = (random.NextSingle() * 2f - 1f) * limit;

            _outputWeightsGradients = new float[_config.VocabSize, _config.HiddenSize];
            _accumulatedOutputWeightsGradients = new float[_config.VocabSize, _config.HiddenSize];
            _outputWeightsOptimizer = new AdamMatrixOptimizer(_config.VocabSize, _config.HiddenSize);
        }

        private float[,] OutputWeightsRef => _config.TieWordEmbeddings ? _embedding.EmbeddingsRef : _outputWeights!;

        public void ZeroGradients()
        {
            EnsureTrainingEnabled();
            _embedding.ZeroGradients();

            foreach (var block in _blocks)
                block.ZeroGradients();

            _finalNorm.ZeroGradients();

            if (!_config.TieWordEmbeddings)
                Matematicas.ParallelClearMatrix(_accumulatedOutputWeightsGradients!);
        }

        public void AverageGradients(int batchSize)
        {
            EnsureTrainingEnabled();
            _embedding.AverageGradients(batchSize);

            foreach (var block in _blocks)
                block.AverageGradients(batchSize);

            _finalNorm.AverageGradients(batchSize);

            if (!_config.TieWordEmbeddings)
            {
                float scale = 1f / batchSize;
                _outputWeightsGradients = TensorOps.Scale(
                    Neuraval.Tensor.Tensor.FromArray2D(_accumulatedOutputWeightsGradients!), scale).ToArray2D();
            }
        }

        public float[,] Forward(int[] inputTokens)
        {
            var (_, logits) = ForwardWithHiddenStates(inputTokens);
            return logits;
        }

        public GqaGenerationCache CreateGenerationCache(int capacity, int batchSize = 1)
        {
            if (capacity <= 0)
                throw new ArgumentException("capacity debe ser positivo");

            if (capacity > _config.MaxPositionEmbeddings)
                throw new ArgumentException(
                    $"capacity ({capacity}) no puede exceder MaxPositionEmbeddings ({_config.MaxPositionEmbeddings})");

            return new GqaGenerationCache(
                _config.NumHiddenLayers,
                batchSize,
                capacity,
                _config.NumKeyValueHeads,
                _config.HeadDim);
        }

        public float[,] ForwardIncremental(int[] newTokens, GqaGenerationCache cache)
        {
            var hidden = ForwardIncrementalHidden(newTokens, cache);
            return ComputeLogits(_finalNorm.Forward(hidden));
        }

        /// <summary>
        /// Processes all tokens into the KV cache but computes logits only for the final token.
        /// This is the output required by autoregressive generation and avoids projecting every
        /// prompt position across the full vocabulary.
        /// </summary>
        public float[] ForwardIncrementalLastToken(int[] newTokens, GqaGenerationCache cache)
        {
            var hidden = ForwardIncrementalHidden(newTokens, cache);
            var lastHidden = new float[1, _config.HiddenSize];
            Buffer.BlockCopy(
                hidden,
                (hidden.GetLength(0) - 1) * _config.HiddenSize * sizeof(float),
                lastHidden,
                0,
                _config.HiddenSize * sizeof(float));

            var logits = ComputeLogits(_finalNorm.Forward(lastHidden));
            var result = new float[_config.VocabSize];
            Buffer.BlockCopy(logits, 0, result, 0, result.Length * sizeof(float));
            return result;
        }

        private float[,] ForwardIncrementalHidden(int[] newTokens, GqaGenerationCache cache)
        {
            if (newTokens == null || newTokens.Length == 0)
                throw new ArgumentException("newTokens no puede ser nulo ni vacío", nameof(newTokens));

            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            if (cache.Layers.Count != _blocks.Count)
                throw new ArgumentException("El cache no tiene el mismo número de capas que el modelo");

            int positionOffset = cache.Length;

            if (positionOffset + newTokens.Length > _config.MaxPositionEmbeddings)
                throw new ArgumentException(
                    $"La secuencia resultante ({positionOffset + newTokens.Length}) excede MaxPositionEmbeddings ({_config.MaxPositionEmbeddings})");

            var embeddings = _embedding.GetEmbeddings(newTokens);
            var hiddenBatch = ToBatch(embeddings);

            for (int i = 0; i < _blocks.Count; i++)
                hiddenBatch = _blocks[i].ForwardIncremental(hiddenBatch, positionOffset, cache.Layers[i]);

            return FromBatch(hiddenBatch);
        }

        private (float[,] hidden, float[,] logits) ForwardWithHiddenStates(int[] inputTokens)
        {
            if (inputTokens.Length > _config.MaxPositionEmbeddings)
            {
                throw new ArgumentException(
                    $"Input sequence length {inputTokens.Length} exceeds maximum {_config.MaxPositionEmbeddings}");
            }

            var embeddings = _embedding.GetEmbeddings(inputTokens);
            var hiddenBatch = ToBatch(embeddings);

            foreach (var block in _blocks)
                hiddenBatch = block.Forward(hiddenBatch);

            var hidden = FromBatch(hiddenBatch);
            hidden = _finalNorm.Forward(hidden);

            var logits = ComputeLogits(hidden);

            return (hidden, logits);
        }

        private float[,] ComputeLogits(float[,] hidden)
        {
            if (_inferenceOnly)
            {
                var hiddenTensorQ8 = Neuraval.Tensor.Tensor.FromArray2D(hidden, DeviceType.Cpu);
                var quantizedOutput = _config.TieWordEmbeddings
                    ? _embedding.QuantizedEmbeddings!
                    : _outputWeightsQ8!;
                return quantizedOutput.Multiply(hiddenTensorQ8).ToArray2D();
            }

            var device = TensorDeviceSelector.Current;
            try
            {
                var hiddenTensor = Neuraval.Tensor.Tensor.FromArray2D(hidden, device);
                var logitsTensor = TensorOps.MatMulTransposeBCachedB(hiddenTensor, OutputWeightsRef, _outputProjectionCache);

                return logitsTensor.ToArray2D();
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return ComputeLogits(hidden);
            }
        }

        private static float[,,] ToBatch(float[,] input)
        {
            int seqLen = input.GetLength(0);
            int dim = input.GetLength(1);
            var batch = new float[1, seqLen, dim];

            System.Buffer.BlockCopy(input, 0, batch, 0, seqLen * dim * sizeof(float));

            return batch;
        }

        private static float[,] FromBatch(float[,,] batch)
        {
            int seqLen = batch.GetLength(1);
            int dim = batch.GetLength(2);
            var output = new float[seqLen, dim];

            System.Buffer.BlockCopy(batch, 0, output, 0, seqLen * dim * sizeof(float));

            return output;
        }

        public float CalculateCausalLoss(int[] sequenceTokens, int lossStartIndex)
        {
            EnsureTrainingEnabled();
            if (sequenceTokens == null || sequenceTokens.Length < 2)
                throw new ArgumentException("sequenceTokens must contain at least two tokens");

            if (lossStartIndex < 0 || lossStartIndex >= sequenceTokens.Length - 1)
                throw new ArgumentException("lossStartIndex must leave at least one token to predict");

            var (hidden, logits) = ForwardWithHiddenStates(sequenceTokens);

            int seqLen = sequenceTokens.Length;
            int hiddenDim = hidden.GetLength(1);
            _pendingHiddenStateGradients = new float[seqLen, hiddenDim];

            var outputWeights = OutputWeightsRef;
            float totalLoss = 0;
            int predictedPositions = 0;

            for (int position = lossStartIndex; position < seqLen - 1; position++)
            {
                int targetToken = sequenceTokens[position + 1];

                if (targetToken < 0 || targetToken >= _config.VocabSize)
                    continue;

                var logitsAtPosition = new float[_config.VocabSize];
                System.Buffer.BlockCopy(
                    logits, position * _config.VocabSize * sizeof(float),
                    logitsAtPosition, 0, _config.VocabSize * sizeof(float));

                var probabilities = Matematicas.ParallelSoftmax(logitsAtPosition);
                totalLoss += -MathF.Log(MathF.Max(probabilities[targetToken], 1e-10f));
                predictedPositions++;

                for (int j = 0; j < _config.VocabSize; j++)
                {
                    float logitGradient = probabilities[j];
                    if (j == targetToken)
                        logitGradient -= 1.0f;

                    for (int k = 0; k < hiddenDim; k++)
                    {
                        AccumulateOutputWeightGradient(j, k, logitGradient * hidden[position, k]);
                        _pendingHiddenStateGradients[position, k] += logitGradient * outputWeights[j, k];
                    }
                }
            }

            if (predictedPositions == 0)
                return 0;

            var gradHidden = _finalNorm.Backward(_pendingHiddenStateGradients);
            var gradHiddenBatch = ToBatch(gradHidden);

            for (int i = _blocks.Count - 1; i >= 0; i--)
                gradHiddenBatch = _blocks[i].Backward(gradHiddenBatch);

            var gradEmbeddings = FromBatch(gradHiddenBatch);
            _embedding.Backward(sequenceTokens, gradEmbeddings);

            return totalLoss / predictedPositions;
        }

        private void AccumulateOutputWeightGradient(int vocabIndex, int hiddenIndex, float value)
        {
            if (_config.TieWordEmbeddings)
            {
                _embedding.AccumulateGradientAt(vocabIndex, hiddenIndex, value);
            }
            else
            {
                _accumulatedOutputWeightsGradients![vocabIndex, hiddenIndex] += value;
            }
        }

        public void UpdateWeights(float learningRate)
        {
            EnsureTrainingEnabled();
            _embedding.UpdateWeights(learningRate);

            foreach (var block in _blocks)
                block.UpdateWeights(learningRate);

            _finalNorm.UpdateWeights(learningRate);

            _outputProjectionCache.Invalidate();

            if (!_config.TieWordEmbeddings)
            {
                _outputWeightsOptimizer!.Update(_outputWeights!, _outputWeightsGradients!, learningRate);
                Matematicas.ParallelClearMatrix(_outputWeightsGradients!);
            }
        }

        public void ResetGradients()
        {
            EnsureTrainingEnabled();
            _embedding.ResetGradients();

            foreach (var block in _blocks)
                block.ResetGradients();

            _finalNorm.ResetGradients();

            if (!_config.TieWordEmbeddings)
            {
                Matematicas.ParallelClearMatrix(_outputWeightsGradients!);
                Matematicas.ParallelClearMatrix(_accumulatedOutputWeightsGradients!);
            }
        }

        public ModernDecoderModelState SaveState()
        {
            EnsureTrainingEnabled();
            var state = new ModernDecoderModelState
            {
                Config = _config.Clone(),
                EmbeddingState = _embedding.SaveState(),
                FinalNormState = _finalNorm.SaveState(),
                BlockStates = new List<ModernDecoderBlockState>()
            };

            foreach (var block in _blocks)
                state.BlockStates.Add(block.SaveState());

            if (!_config.TieWordEmbeddings)
            {
                state.OutputWeights = FlattenMatrix(_outputWeights!);
                state.OutputWeightsOptimizerState = _outputWeightsOptimizer!.SaveState();
            }

            return state;
        }

        /// <summary>Loads a model from state; inference-only mode omits training buffers and consumes large flattened load buffers.</summary>
        public static ModernDecoderModel LoadState(ModernDecoderModelState state, bool inferenceOnly = false)
        {
            var model = new ModernDecoderModel(state.Config, seed: 42, inferenceOnly: inferenceOnly);

            model._embedding = EmbeddingLayer.LoadState(state.EmbeddingState, inferenceOnly);
            if (inferenceOnly)
                state.EmbeddingState.Embeddings = Array.Empty<float>();
            model._finalNorm = RMSNorm.LoadState(state.FinalNormState, inferenceOnly);

            model._blocks = new List<ModernDecoderBlock>();
            foreach (var blockState in state.BlockStates)
            {
                model._blocks.Add(ModernDecoderBlock.LoadState(blockState, inferenceOnly));
                if (inferenceOnly)
                {
                    // Feed-forward state is flattened; its loaded matrices now own the data.
                    // Drop the temporary arrays promptly so the GC can reclaim them between layers.
                    blockState.FeedforwardState.WeightsGate = Array.Empty<float>();
                    blockState.FeedforwardState.WeightsUp = Array.Empty<float>();
                    blockState.FeedforwardState.WeightsDown = Array.Empty<float>();
                    // Attention matrices were requantized into the runtime Q8 representation.
                    blockState.AttentionState.Wq = new float[0, 0];
                    blockState.AttentionState.Wk = new float[0, 0];
                    blockState.AttentionState.Wv = new float[0, 0];
                    blockState.AttentionState.Wo = new float[0, 0];
                }
            }

            if (!state.Config.TieWordEmbeddings && state.OutputWeights != null)
            {
                if (inferenceOnly)
                {
                    model._outputWeightsQ8 = QuantizedMatrixQ8.FromOutputMajor(
                        state.OutputWeights, state.Config.VocabSize, state.Config.HiddenSize);
                    state.OutputWeights = null;
                }
                else
                {
                    model._outputWeights = UnflattenMatrix(state.OutputWeights, state.Config.VocabSize, state.Config.HiddenSize);
                }

                if (!inferenceOnly && state.OutputWeightsOptimizerState != null)
                    model._outputWeightsOptimizer!.LoadStateInto(state.OutputWeightsOptimizerState);
            }

            model._outputProjectionCache.Invalidate();

            return model;
        }

        public bool IsInferenceOnly => _inferenceOnly;

        private void EnsureTrainingEnabled()
        {
            if (_inferenceOnly)
                throw new InvalidOperationException("Este modelo se cargó en modo de inferencia y no conserva gradientes ni estado de entrenamiento.");
        }

        private static float[] FlattenMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows * cols];

            System.Buffer.BlockCopy(matrix, 0, result, 0, result.Length * sizeof(float));

            return result;
        }

        private static float[,] UnflattenMatrix(float[] array, int rows, int cols)
        {
            var matrix = new float[rows, cols];

            System.Buffer.BlockCopy(array, 0, matrix, 0, array.Length * sizeof(float));

            return matrix;
        }
    }

    public class ModernDecoderModelState
    {
        public TransformerConfig Config { get; set; } = null!;
        public EmbeddingLayerState EmbeddingState { get; set; } = null!;
        public RMSNormState FinalNormState { get; set; } = null!;
        public List<ModernDecoderBlockState> BlockStates { get; set; } = new List<ModernDecoderBlockState>();
        public float[]? OutputWeights { get; set; }
        public AdamMatrixOptimizerState? OutputWeightsOptimizerState { get; set; }
    }
}
