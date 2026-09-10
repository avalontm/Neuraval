using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using Neuraval.Abstractions;
using Neuraval.Core.Utils;
using Neuraval.Cuda;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class TransformerModel : ITrainableModel
    {
        private readonly int _vocabSize;
        private readonly int _embeddingDim;
        private readonly int _numLayers;
        private readonly int _numHeads;
        private readonly int _feedforwardDim;
        private readonly int _maxSequenceLength;
        private readonly float _dropout;

        private EmbeddingLayer _embedding;
        private PositionalEncoding _positionalEncoding;
        private List<TransformerBlock> _blocks;
        private LayerNormalization _finalNorm;
        private float[] _outputBias;

        private float[] _outputBiasGradients;
        private float[,]? _pendingHiddenStateGradients;
        private float[] _accumulatedOutputBiasGradients;
        private AdamVectorOptimizer _outputBiasOptimizer = null!;
        private CudaWeightCache _outputProjectionCache = null!;

        /// <summary>
        /// Cuando hay LoRA habilitado con congelamiento de base (Fase 5.5),
        /// <see cref="UpdateWeights"/> deja de tocar el embedding, la norma
        /// final y el bias de salida — solo se entrenan los adaptadores LoRA
        /// dentro de cada <see cref="TransformerBlock"/>.
        /// </summary>
        private bool _freezeNonLoraWeights;

        public float[,]? PendingHiddenStateGradients => _pendingHiddenStateGradients;

        public int LastPredictedPositions { get; private set; }

        public int VocabSize => _vocabSize;
        public int EmbeddingDim => _embeddingDim;
        public int MaxSequenceLength => _maxSequenceLength;
        public bool HasLora => _blocks.Count > 0 && _blocks[0].HasLora;

        public TransformerModel(
            int vocabSize,
            int embeddingDim,
            int numLayers,
            int numHeads,
            int feedforwardDim,
            int maxSequenceLength,
            float dropout = 0.1f,
            int seed = 42)
        {
            _vocabSize = vocabSize;
            _embeddingDim = embeddingDim;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedforwardDim = feedforwardDim;
            _maxSequenceLength = maxSequenceLength;
            _dropout = dropout;

            _embedding = new EmbeddingLayer(vocabSize, embeddingDim, seed);
            _positionalEncoding = new PositionalEncoding(maxSequenceLength, embeddingDim);

            _blocks = new List<TransformerBlock>();
            for (int i = 0; i < numLayers; i++)
            {
                _blocks.Add(new TransformerBlock(embeddingDim, numHeads, feedforwardDim, dropout, seed + i + 1));
            }

            _finalNorm = new LayerNormalization(embeddingDim);

            InitializeOutputLayer();
        }

        private void InitializeOutputLayer()
        {
            _outputBias = new float[_vocabSize];
            _outputBiasGradients = new float[_vocabSize];
            _accumulatedOutputBiasGradients = new float[_vocabSize];
            _outputBiasOptimizer = new AdamVectorOptimizer(_vocabSize);
            _outputProjectionCache = new CudaWeightCache(_vocabSize, _embeddingDim);
        }

        /// <summary>
        /// Habilita adaptadores LoRA (Fase 5.5) en la capa de atención de
        /// cada bloque del modelo. Si <paramref name="freezeBase"/> es true
        /// (default), además congela todo lo demás — embedding, bloques
        /// completos salvo los adaptadores, norma final y bias de salida —
        /// de forma que <see cref="UpdateWeights"/> solo entrene A/B de LoRA.
        /// No hace nada si el modelo ya tenía LoRA habilitado.
        /// </summary>
        public void EnableLora(int rank, float alpha, bool freezeBase = true, int seed = 9001)
        {
            if (HasLora) return;

            for (int i = 0; i < _blocks.Count; i++)
            {
                _blocks[i].EnableLora(rank, alpha, seed + i * 10);
                _blocks[i].SetFreezeBaseWeights(freezeBase);
            }

            _freezeNonLoraWeights = freezeBase;
        }

        /// <summary>
        /// Guarda únicamente los adaptadores LoRA de todos los bloques (y el
        /// flag de congelamiento), pensado para exportar/importar por
        /// separado del modelo base con el formato <c>.navlora</c>. Devuelve
        /// <c>null</c> si el modelo no tiene LoRA habilitado.
        /// </summary>
        public TransformerModelLoraState? SaveLoraState()
        {
            if (!HasLora) return null;

            var state = new TransformerModelLoraState
            {
                NumLayers = _numLayers,
                EmbeddingDim = _embeddingDim,
                FreezeNonLoraWeights = _freezeNonLoraWeights
            };

            foreach (var block in _blocks)
            {
                var blockLoraState = block.SaveLoraState()
                    ?? throw new InvalidOperationException("Bloque sin adaptadores LoRA en un modelo que reporta HasLora=true");
                state.BlockStates.Add(blockLoraState);
            }

            return state;
        }

        /// <summary>
        /// Carga adaptadores LoRA previamente exportados (por ejemplo desde un
        /// archivo <c>.navlora</c>) sobre este modelo. Habilita LoRA en cada
        /// bloque si todavía no estaba habilitado.
        /// </summary>
        public void LoadLoraState(TransformerModelLoraState state)
        {
            if (state.EmbeddingDim != _embeddingDim || state.NumLayers != _numLayers)
            {
                throw new ArgumentException(
                    $"El adaptador LoRA ({state.NumLayers} capas, embeddingDim={state.EmbeddingDim}) " +
                    $"no es compatible con este modelo ({_numLayers} capas, embeddingDim={_embeddingDim}).");
            }

            if (state.BlockStates.Count != _blocks.Count)
            {
                throw new ArgumentException(
                    $"El adaptador LoRA trae {state.BlockStates.Count} bloques pero el modelo tiene {_blocks.Count}.");
            }

            for (int i = 0; i < _blocks.Count; i++)
            {
                _blocks[i].LoadLoraState(state.BlockStates[i]);
            }

            _freezeNonLoraWeights = state.FreezeNonLoraWeights;
        }

        public void ZeroGradients()
        {
            Array.Clear(_accumulatedOutputBiasGradients, 0, _accumulatedOutputBiasGradients.Length);

            _embedding.ZeroGradients();
            foreach (var block in _blocks)
            {
                block.ZeroGradients();
            }
            _finalNorm.ZeroGradients();
        }

        public void AverageGradients(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive");

            float scale = 1.0f / batchSize;

            Parallel.For(0, _vocabSize, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                _outputBiasGradients[i] = _accumulatedOutputBiasGradients[i] * scale;
            });

            _embedding.AverageGradients(batchSize);
            foreach (var block in _blocks)
            {
                block.AverageGradients(batchSize);
            }
            _finalNorm.AverageGradients(batchSize);
        }

        public float ClipGradients(float maxNorm)
        {
            float gradNorm = CalculateGlobalGradientNorm();

            if (gradNorm > maxNorm)
            {
                float scale = maxNorm / (gradNorm + 1e-10f);

                Parallel.For(0, _vocabSize, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
                {
                    _outputBiasGradients[i] *= scale;
                });

                _embedding.ScaleGradients(scale);
                foreach (var block in _blocks)
                {
                    block.ScaleGradients(scale);
                }
                _finalNorm.ScaleGradients(scale);
            }

            return gradNorm;
        }

        private float CalculateGlobalGradientNorm()
        {
            float sumSquared = 0;

            for (int i = 0; i < _vocabSize; i++)
            {
                sumSquared += _outputBiasGradients[i] * _outputBiasGradients[i];
            }

            sumSquared += _embedding.SumSquaredGradients();

            foreach (var block in _blocks)
            {
                sumSquared += block.SumSquaredGradients();
            }

            sumSquared += _finalNorm.SumSquaredGradients();

            return MathF.Sqrt(sumSquared);
        }

        public float[,] Forward(int[] inputTokens, bool training = true)
        {
            var (hidden, logits) = ForwardWithHiddenStates(inputTokens, training);
            return logits;
        }

        private (float[,] hidden, float[,] logits) ForwardWithHiddenStates(int[] inputTokens, bool training)
        {
            if (inputTokens.Length > _maxSequenceLength)
            {
                throw new ArgumentException($"Input sequence length {inputTokens.Length} exceeds maximum {_maxSequenceLength}");
            }

            var embeddings = _embedding.GetEmbeddings(inputTokens);
            var embeddingsWithPosition = _positionalEncoding.AddToEmbeddings(embeddings);

            var hidden = embeddingsWithPosition;
            var causalMask = BuildCausalMask(inputTokens.Length);

            foreach (var block in _blocks)
            {
                hidden = block.Forward(hidden, causalMask, training);
            }

            hidden = _finalNorm.Forward(hidden);

            int seqLen = hidden.GetLength(0);
            var logits = ComputeLogits(hidden, seqLen);

            return (hidden, logits);
        }

        private float[,] ComputeLogits(float[,] hidden, int seqLen)
        {
            var device = TensorDeviceSelector.Current;

            try
            {
                var hiddenTensor = Neuraval.Tensor.Tensor.FromArray2D(hidden, device);
                var logitsTensor = TensorOps.MatMulTransposeBCachedB(hiddenTensor, _embedding.EmbeddingsRef, _outputProjectionCache);
                var logits = new float[seqLen, _vocabSize];

                Parallel.For(0, seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
                {
                    int rowOffset = i * _vocabSize;

                    for (int j = 0; j < _vocabSize; j++)
                    {
                        logits[i, j] = logitsTensor.Buffer[rowOffset + j] + _outputBias[j];
                    }
                });

                return logits;
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return ComputeLogits(hidden, seqLen);
            }
        }

        private float[,] BuildCausalMask(int sequenceLength)
        {
            var mask = new float[sequenceLength, sequenceLength];

            for (int i = 0; i < sequenceLength; i++)
            {
                for (int j = 0; j < sequenceLength; j++)
                {
                    mask[i, j] = j <= i ? 1.0f : 0.0f;
                }
            }

            return mask;
        }

        public float[] Predict(int[] inputTokens)
        {
            var logits = Forward(inputTokens, false);
            int lastPosition = logits.GetLength(0) - 1;

            var lastLogits = new float[_vocabSize];
            Buffer.BlockCopy(logits, lastPosition * _vocabSize * sizeof(float), lastLogits, 0, _vocabSize * sizeof(float));

            return Matematicas.ParallelSoftmax(lastLogits);
        }

        public int PredictNextToken(int[] inputTokens)
        {
            var probabilities = Predict(inputTokens);
            return ArgMax(probabilities);
        }

        public string GenerateText(int[] seedTokens, int maxLength, int endToken)
        {
            var tokens = new List<int>(seedTokens);

            for (int i = 0; i < maxLength; i++)
            {
                int nextToken = PredictNextToken(tokens.ToArray());

                if (nextToken == endToken)
                {
                    break;
                }

                tokens.Add(nextToken);

                if (tokens.Count >= _maxSequenceLength)
                {
                    break;
                }
            }

            return string.Join(",", tokens);
        }

        public GenerationCache CreateGenerationCache()
        {
            return new GenerationCache(_numLayers, _maxSequenceLength, _embeddingDim);
        }

        private float[] PredictNextIncremental(int[] newTokens, GenerationCache cache, int position)
        {
            if (position + newTokens.Length > _maxSequenceLength)
            {
                throw new ArgumentException(
                    $"La posición {position + newTokens.Length - 1} excede el máximo {_maxSequenceLength - 1}");
            }

            var embeddings = _embedding.GetEmbeddings(newTokens);
            var hidden = _positionalEncoding.AddToEmbeddingsAtOffset(embeddings, position);

            for (int i = 0; i < _blocks.Count; i++)
            {
                hidden = _blocks[i].ForwardIncremental(hidden, cache.Layers[i]);
            }

            hidden = _finalNorm.Forward(hidden);

            int lastRow = hidden.GetLength(0) - 1;
            var lastHidden = new float[1, _embeddingDim];
            Buffer.BlockCopy(hidden, lastRow * _embeddingDim * sizeof(float), lastHidden, 0, _embeddingDim * sizeof(float));

            var logits = ComputeLogits(lastHidden, 1);
            var lastLogits = new float[_vocabSize];
            Buffer.BlockCopy(logits, 0, lastLogits, 0, _vocabSize * sizeof(float));

            return Matematicas.ParallelSoftmax(lastLogits);
        }

        public string GenerateTextCached(int[] seedTokens, int maxLength, int endToken)
        {
            if (seedTokens == null || seedTokens.Length == 0)
            {
                throw new ArgumentException("seedTokens no puede estar vacío");
            }

            if (seedTokens.Length > _maxSequenceLength)
            {
                throw new ArgumentException($"Seed sequence length {seedTokens.Length} exceeds maximum {_maxSequenceLength}");
            }

            var cache = CreateGenerationCache();
            var tokens = new List<int>(seedTokens);

            var probabilities = PredictNextIncremental(seedTokens, cache, 0);

            for (int i = 0; i < maxLength; i++)
            {
                int nextToken = ArgMax(probabilities);

                if (nextToken == endToken)
                {
                    break;
                }

                tokens.Add(nextToken);

                if (tokens.Count >= _maxSequenceLength)
                {
                    break;
                }

                probabilities = PredictNextIncremental(new[] { nextToken }, cache, cache.Length);
            }

            return string.Join(",", tokens);
        }

        private int ArgMax(float[] array)
        {
            int maxIndex = 0;
            float maxValue = array[0];

            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] > maxValue)
                {
                    maxValue = array[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        public float CalculateLoss(int[] inputTokens, int[] targetTokens)
        {
            if (inputTokens == null || inputTokens.Length == 0)
                throw new ArgumentException("inputTokens cannot be null or empty");

            if (targetTokens == null || targetTokens.Length == 0)
                throw new ArgumentException("targetTokens cannot be null or empty");

            var logits = Forward(inputTokens, true);

            int logitsSeqLen = logits.GetLength(0);
            int targetsLen = targetTokens.Length;
            int seqLen = Math.Min(logitsSeqLen, targetsLen);

            if (seqLen == 0)
                throw new InvalidOperationException("Sequence length is zero");

            float totalLoss = 0;

            for (int i = 0; i < seqLen; i++)
            {
                var logitsAtPos = new float[_vocabSize];
                Buffer.BlockCopy(logits, i * _vocabSize * sizeof(float), logitsAtPos, 0, _vocabSize * sizeof(float));

                var probs = Matematicas.ParallelSoftmax(logitsAtPos);

                int targetToken = targetTokens[i];
                if (targetToken >= 0 && targetToken < _vocabSize)
                {
                    totalLoss += -MathF.Log(MathF.Max(probs[targetToken], 1e-10f));

                    for (int j = 0; j < _vocabSize; j++)
                    {
                        float gradient = probs[j];
                        if (j == targetToken)
                        {
                            gradient -= 1.0f;
                        }

                        for (int k = 0; k < _embeddingDim; k++)
                        {
                            _embedding.AccumulateGradientAt(j, k, gradient);
                        }
                        _accumulatedOutputBiasGradients[j] += gradient;
                    }
                }
                else
                {
                    Console.WriteLine($"Warning: Invalid target token {targetToken} at position {i}");
                }
            }

            return totalLoss / seqLen;
        }

        public float CalculateCausalLoss(int[] sequenceTokens, int lossStartIndex)
        {
            if (sequenceTokens == null || sequenceTokens.Length < 2)
                throw new ArgumentException("sequenceTokens must contain at least two tokens");

            if (lossStartIndex < 0 || lossStartIndex >= sequenceTokens.Length - 1)
                throw new ArgumentException("lossStartIndex must leave at least one token to predict");

            var (hidden, logits) = ForwardWithHiddenStates(sequenceTokens, true);

            int seqLen = sequenceTokens.Length;
            int hiddenDim = hidden.GetLength(1);
            _pendingHiddenStateGradients = new float[seqLen, hiddenDim];

            float totalLoss = 0;
            int predictedPositions = 0;

            for (int position = lossStartIndex; position < seqLen - 1; position++)
            {
                int targetToken = sequenceTokens[position + 1];

                if (targetToken < 0 || targetToken >= _vocabSize)
                {
                    Console.WriteLine($"Warning: Invalid target token {targetToken} at position {position + 1}");
                    continue;
                }

                var logitsAtPosition = new float[_vocabSize];
                Buffer.BlockCopy(logits, position * _vocabSize * sizeof(float), logitsAtPosition, 0, _vocabSize * sizeof(float));

                var probabilities = Matematicas.ParallelSoftmax(logitsAtPosition);
                totalLoss += -MathF.Log(MathF.Max(probabilities[targetToken], 1e-10f));
                predictedPositions++;

                for (int j = 0; j < _vocabSize; j++)
                {
                    float logitGradient = probabilities[j];
                    if (j == targetToken)
                    {
                        logitGradient -= 1.0f;
                    }

                    for (int k = 0; k < hiddenDim; k++)
                    {
                        _embedding.AccumulateGradientAt(j, k, logitGradient * hidden[position, k]);
                        _pendingHiddenStateGradients[position, k] += logitGradient * _embedding.EmbeddingsRef[j, k];
                    }

                    _accumulatedOutputBiasGradients[j] += logitGradient;
                }
            }

            if (predictedPositions == 0)
            {
                return 0;
            }

            var pendingHiddenStateGradients = _pendingHiddenStateGradients
                ?? throw new InvalidOperationException("_pendingHiddenStateGradients no fue inicializado");
            var gradHidden = _finalNorm.Backward(pendingHiddenStateGradients, 0.0f);

            for (int i = _blocks.Count - 1; i >= 0; i--)
            {
                gradHidden = _blocks[i].Backward(gradHidden);
            }

            _embedding.Backward(sequenceTokens, gradHidden);

            return totalLoss / predictedPositions;
        }

        private float[,] BuildCausalPaddingMask(int sequenceLength, int validLength)
        {
            var mask = new float[sequenceLength, sequenceLength];

            for (int i = 0; i < sequenceLength; i++)
            {
                for (int j = 0; j < sequenceLength; j++)
                {
                    mask[i, j] = (j <= i && j < validLength) ? 1.0f : 0.0f;
                }
            }

            return mask;
        }

        private float[,,] BuildCausalPaddingMaskBatch(int sequenceLength, int[] validLengths)
        {
            int batchSize = validLengths.Length;
            var maskBatch = new float[batchSize, sequenceLength, sequenceLength];

            for (int b = 0; b < batchSize; b++)
            {
                var sampleMask = BuildCausalPaddingMask(sequenceLength, validLengths[b]);
                Matematicas.SetBatchSlice(maskBatch, b, sampleMask);
            }

            return maskBatch;
        }

        private (float[,,] hidden, float[,,] logits) ForwardBatchWithHiddenStates(
            int[,] tokenBatch, int[] validLengths, bool training)
        {
            int batchSize = tokenBatch.GetLength(0);
            int seqLen = tokenBatch.GetLength(1);

            if (seqLen > _maxSequenceLength)
            {
                throw new ArgumentException($"Input sequence length {seqLen} exceeds maximum {_maxSequenceLength}");
            }

            if (validLengths.Length != batchSize)
            {
                throw new ArgumentException("validLengths debe tener un elemento por cada secuencia del batch");
            }

            var embeddings = _embedding.GetEmbeddingsBatch(tokenBatch);
            var embeddingsWithPosition = _positionalEncoding.AddToEmbeddingsBatch(embeddings);

            var maskBatch = BuildCausalPaddingMaskBatch(seqLen, validLengths);

            var hidden = embeddingsWithPosition;
            foreach (var block in _blocks)
            {
                hidden = block.ForwardBatch(hidden, maskBatch, training);
            }

            hidden = _finalNorm.ForwardBatch(hidden);

            var logits = ComputeLogitsBatch(hidden, batchSize, seqLen);

            return (hidden, logits);
        }

        private float[,,] ComputeLogitsBatch(float[,,] hidden, int batchSize, int seqLen)
        {
            var device = TensorDeviceSelector.Current;

            try
            {
                var flatHidden = FlattenBatch(hidden);
                var hiddenTensor = Neuraval.Tensor.Tensor.FromArray2D(flatHidden, device);
                var logitsTensor = TensorOps.MatMulTransposeBCachedB(hiddenTensor, _embedding.EmbeddingsRef, _outputProjectionCache);
                var logits = new float[batchSize, seqLen, _vocabSize];

                Parallel.For(0, batchSize * seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, idx =>
                {
                    int b = idx / seqLen;
                    int i = idx % seqLen;
                    int rowOffset = idx * _vocabSize;

                    for (int j = 0; j < _vocabSize; j++)
                    {
                        logits[b, i, j] = logitsTensor.Buffer[rowOffset + j] + _outputBias[j];
                    }
                });

                return logits;
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return ComputeLogitsBatch(hidden, batchSize, seqLen);
            }
        }

        private static float[,] FlattenBatch(float[,,] batch)
        {
            int batchSize = batch.GetLength(0);
            int seqLen = batch.GetLength(1);
            int dim = batch.GetLength(2);
            var flat = new float[batchSize * seqLen, dim];

            // batch[b, i, j] y flat[b*seqLen + i, j] son el mismo layout row-major
            // contiguo en memoria (misma cantidad total de elementos, mismo orden),
            // así que aplanar es un único memcpy en vez de una copia elemento a elemento.
            Buffer.BlockCopy(batch, 0, flat, 0, batchSize * seqLen * dim * sizeof(float));

            return flat;
        }

        public float[,,] ForwardBatch(int[,] tokenBatch, int[] validLengths, bool training = true)
        {
            var (_, logits) = ForwardBatchWithHiddenStates(tokenBatch, validLengths, training);
            return logits;
        }

        public float CalculateCausalLossBatch(int[,] sequenceBatch, int[] validLengths, int[] lossStartIndices)
        {
            if (sequenceBatch == null)
                throw new ArgumentException("sequenceBatch no puede ser null");

            int batchSize = sequenceBatch.GetLength(0);
            int seqLen = sequenceBatch.GetLength(1);

            if (validLengths == null || validLengths.Length != batchSize)
                throw new ArgumentException("validLengths debe tener un elemento por cada secuencia del batch");

            if (lossStartIndices == null || lossStartIndices.Length != batchSize)
                throw new ArgumentException("lossStartIndices debe tener un elemento por cada secuencia del batch");

            for (int b = 0; b < batchSize; b++)
            {
                if (validLengths[b] < 2 || validLengths[b] > seqLen)
                {
                    throw new ArgumentException(
                        $"validLengths[{b}]={validLengths[b]} debe estar entre 2 y seqLen={seqLen}");
                }

                if (lossStartIndices[b] < 0 || lossStartIndices[b] >= validLengths[b] - 1)
                {
                    throw new ArgumentException(
                        $"lossStartIndices[{b}]={lossStartIndices[b]} debe dejar al menos un token por predecir " +
                        $"dentro de validLengths[{b}]={validLengths[b]}");
                }
            }

            var (hidden, logits) = ForwardBatchWithHiddenStates(sequenceBatch, validLengths, true);

            var pendingHiddenGradBatch = new float[batchSize, seqLen, _embeddingDim];

            float totalLoss = 0;
            int predictedPositions = 0;

            for (int b = 0; b < batchSize; b++)
            {
                for (int position = lossStartIndices[b]; position < validLengths[b] - 1; position++)
                {
                    int targetToken = sequenceBatch[b, position + 1];

                    if (targetToken < 0 || targetToken >= _vocabSize)
                    {
                        Console.WriteLine($"Warning: Invalid target token {targetToken} at batch {b}, position {position + 1}");
                        continue;
                    }

                    var logitsAtPosition = new float[_vocabSize];
                    Buffer.BlockCopy(logits, (b * seqLen + position) * _vocabSize * sizeof(float), logitsAtPosition, 0, _vocabSize * sizeof(float));

                    var probabilities = Matematicas.ParallelSoftmax(logitsAtPosition);
                    totalLoss += -MathF.Log(MathF.Max(probabilities[targetToken], 1e-10f));
                    predictedPositions++;

                    for (int j = 0; j < _vocabSize; j++)
                    {
                        float logitGradient = probabilities[j];
                        if (j == targetToken)
                        {
                            logitGradient -= 1.0f;
                        }

                        for (int k = 0; k < _embeddingDim; k++)
                        {
                            _embedding.AccumulateGradientAt(j, k, logitGradient * hidden[b, position, k]);
                            pendingHiddenGradBatch[b, position, k] += logitGradient * _embedding.EmbeddingsRef[j, k];
                        }

                        _accumulatedOutputBiasGradients[j] += logitGradient;
                    }
                }
            }

            LastPredictedPositions = predictedPositions;

            if (predictedPositions == 0)
            {
                return 0;
            }

            var gradHidden = _finalNorm.BackwardBatch(pendingHiddenGradBatch, 0.0f);

            for (int i = _blocks.Count - 1; i >= 0; i--)
            {
                gradHidden = _blocks[i].BackwardBatch(gradHidden);
            }

            _embedding.BackwardBatch(sequenceBatch, gradHidden);

            return totalLoss / predictedPositions;
        }

        public void UpdateWeights(float learningRate)
        {
            if (!_freezeNonLoraWeights)
            {
                _embedding.UpdateWeights(learningRate);
            }

            foreach (var block in _blocks)
            {
                block.UpdateWeights(learningRate);
            }

            if (!_freezeNonLoraWeights)
            {
                _finalNorm.UpdateWeights(learningRate);

                _outputBiasOptimizer.Update(_outputBias, _outputBiasGradients, learningRate);
            }

            Array.Clear(_outputBiasGradients, 0, _outputBiasGradients.Length);

            _outputProjectionCache.Invalidate();
        }

        public TransformerModelState SaveState()
        {
            var state = new TransformerModelState
            {
                VocabSize = _vocabSize,
                EmbeddingDim = _embeddingDim,
                NumLayers = _numLayers,
                NumHeads = _numHeads,
                FeedforwardDim = _feedforwardDim,
                MaxSequenceLength = _maxSequenceLength,
                Dropout = _dropout,
                EmbeddingState = _embedding.SaveState(),
                BlockStates = _blocks.Select(b => b.SaveState()).ToList(),
                FinalNormState = _finalNorm.SaveState(),
                OutputBias = (float[])_outputBias.Clone(),
                OutputBiasOptimizerState = _outputBiasOptimizer.SaveState()
            };

            return state;
        }

        public static TransformerModel LoadState(TransformerModelState state)
        {
            var model = new TransformerModel(
                state.VocabSize,
                state.EmbeddingDim,
                state.NumLayers,
                state.NumHeads,
                state.FeedforwardDim,
                state.MaxSequenceLength,
                state.Dropout);

            model.RestoreFrom(state);

            return model;
        }

        public void RestoreFrom(TransformerModelState state)
        {
            _embedding = EmbeddingLayer.LoadState(state.EmbeddingState);

            _blocks.Clear();
            foreach (var blockState in state.BlockStates)
            {
                _blocks.Add(TransformerBlock.LoadState(blockState));
            }

            _finalNorm = LayerNormalization.LoadState(state.FinalNormState);
            _outputBias = (float[])state.OutputBias.Clone();

            if (state.OutputBiasOptimizerState != null) _outputBiasOptimizer.LoadStateInto(state.OutputBiasOptimizerState);

            // Si los bloques trajeron adaptadores LoRA con la base congelada,
            // el modelo entero se considera en modo "solo LoRA" al recargarlo.
            _freezeNonLoraWeights = _blocks.Count > 0 && _blocks[0].HasLora && _blocks[0].IsBaseFrozen;

            _outputProjectionCache.Invalidate();
        }

    }

    public class TransformerModelState
    {
        public int VocabSize { get; set; }
        public int EmbeddingDim { get; set; }
        public int NumLayers { get; set; }
        public int NumHeads { get; set; }
        public int FeedforwardDim { get; set; }
        public int MaxSequenceLength { get; set; }
        public float Dropout { get; set; }
        public EmbeddingLayerState EmbeddingState { get; set; }
        public List<TransformerBlockState> BlockStates { get; set; }
        public LayerNormalizationState FinalNormState { get; set; }
        public float[] OutputBias { get; set; }
        public AdamVectorOptimizerState? OutputBiasOptimizerState { get; set; }

        public TransformerModelState()
        {
            EmbeddingState = new EmbeddingLayerState();
            BlockStates = new List<TransformerBlockState>();
            FinalNormState = new LayerNormalizationState();
            OutputBias = Array.Empty<float>();
        }
    }
}