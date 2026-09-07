using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using Neuraval.Abstractions;
using Neuraval.Core.Utils;

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

        public float[,]? PendingHiddenStateGradients => _pendingHiddenStateGradients;

        public int LastPredictedPositions { get; private set; }

        public int VocabSize => _vocabSize;
        public int EmbeddingDim => _embeddingDim;
        public int MaxSequenceLength => _maxSequenceLength;

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
            float gradNorm = CalculateGradientNorm();

            if (gradNorm > maxNorm)
            {
                float scale = maxNorm / (gradNorm + 1e-10f);

                Parallel.For(0, _vocabSize, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
                {
                    _outputBiasGradients[i] *= scale;
                });

                _embedding.ClipGradients(maxNorm);
                foreach (var block in _blocks)
                {
                    block.ClipGradients(maxNorm);
                }
                _finalNorm.ClipGradients(maxNorm);
            }

            return gradNorm;
        }

        private float CalculateGradientNorm()
        {
            float sumSquared = 0;

            for (int i = 0; i < _vocabSize; i++)
            {
                sumSquared += _outputBiasGradients[i] * _outputBiasGradients[i];
            }

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
            var logits = new float[seqLen, _vocabSize];

            Parallel.For(0, seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                for (int j = 0; j < _vocabSize; j++)
                {
                    float sum = _outputBias[j];
                    for (int k = 0; k < _embeddingDim; k++)
                    {
                        sum += hidden[i, k] * _embedding.EmbeddingsRef[j, k];
                    }
                    logits[i, j] = sum;
                }
            });

            return (hidden, logits);
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
            for (int i = 0; i < _vocabSize; i++)
            {
                lastLogits[i] = logits[lastPosition, i];
            }

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
                for (int j = 0; j < _vocabSize; j++)
                {
                    logitsAtPos[j] = logits[i, j];
                }

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
                for (int j = 0; j < _vocabSize; j++)
                {
                    logitsAtPosition[j] = logits[position, j];
                }

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

        /// <summary>
        /// Construye la máscara combinada causal+padding para UNA secuencia: además de la
        /// restricción causal de siempre (j &lt;= i), bloquea también cualquier posición de key
        /// j &gt;= validLength (relleno de <c>PadToken</c>). Ver la nota extensa en
        /// <see cref="BuildCausalPaddingMaskBatch"/> sobre por qué esta restricción extra es, en la
        /// práctica, redundante con el padding a la derecha + la máscara causal para las filas de
        /// consulta reales — se implementa de todos modos por robustez explícita, siguiendo el
        /// pedido literal del plan (Fase 2 → diferido a Fase 4.4): "máscara de padding combinada
        /// con la causal, excluida del cálculo de loss".
        /// </summary>
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

        /// <summary>
        /// Versión por batch de <see cref="BuildCausalPaddingMask"/>: una máscara distinta por
        /// elemento, porque <c>validLengths</c> difiere entre ejemplos del mismo batch (secuencias
        /// de distinta longitud real, todas rellenadas con <c>PadToken</c> hasta la misma
        /// <c>sequenceLength</c> por el llamador — ver <c>SupervisedTrainer.BuildPaddedCausalBatch</c>).
        ///
        /// Nota sobre por qué esto es, en rigor, redundante para las posiciones que sí importan:
        /// con padding a la derecha (el relleno siempre va DESPUÉS del contenido real, nunca antes)
        /// y máscara causal ya correcta (Fase 1), cualquier posición de consulta real i &lt;
        /// validLength solo puede atender a keys j &lt;= i &lt; validLength — es decir, la propia
        /// máscara causal ya excluye el padding para esas filas, sin necesitar la condición extra
        /// "j &lt; validLength". Esa condición solo cambia algo para las filas de consulta que ELLAS
        /// MISMAS son padding (i &gt;= validLength), y esas filas nunca se usan: no participan del
        /// loss (el loop de <see cref="CalculateCausalLossBatch"/> se detiene en
        /// <c>validLengths[b] - 1</c>) y por lo tanto reciben gradiente cero desde arriba, lo que
        /// (como se explica en <c>FASE_4_4_CAMBIOS.md</c>) hace que su contribución a los gradientes
        /// de los pesos compartidos (Q/K/V, feedforward, layer norm) sea exactamente cero sin
        /// importar qué hayan calculado en el forward. Se implementa la máscara completa de todos
        /// modos porque es el comportamiento explícito y auditable que pide el plan, y porque deja
        /// de depender de este razonamiento si en el futuro se cambia a padding por la izquierda.
        /// </summary>
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

        /// <summary>
        /// Forward por batch con estados ocultos expuestos, análogo a
        /// <see cref="ForwardWithHiddenStates"/> pero para un batch real (todas las capas usan sus
        /// métodos <c>*Batch</c> desde la Fase 4.1-4.3). <paramref name="tokenBatch"/> debe venir ya
        /// rellenado (mismo largo de secuencia para todos los elementos, relleno con
        /// <c>PadToken</c>) y <paramref name="validLengths"/> indica cuántos tokens de cada fila son
        /// reales (el resto es padding).
        /// </summary>
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

            var logits = new float[batchSize, seqLen, _vocabSize];

            Parallel.For(0, batchSize * seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, idx =>
            {
                int b = idx / seqLen;
                int i = idx % seqLen;

                for (int j = 0; j < _vocabSize; j++)
                {
                    float sum = _outputBias[j];
                    for (int k = 0; k < _embeddingDim; k++)
                    {
                        sum += hidden[b, i, k] * _embedding.EmbeddingsRef[j, k];
                    }
                    logits[b, i, j] = sum;
                }
            });

            return (hidden, logits);
        }

        /// <summary>
        /// Forward por batch expuesto públicamente (logits únicamente), análogo a
        /// <see cref="Forward"/> pero para varias secuencias a la vez con padding. Útil para
        /// inferencia por batch y para tests de equivalencia batch-vs-loop; el entrenamiento real
        /// pasa por <see cref="CalculateCausalLossBatch"/>, que reutiliza el mismo forward interno.
        /// </summary>
        public float[,,] ForwardBatch(int[,] tokenBatch, int[] validLengths, bool training = true)
        {
            var (_, logits) = ForwardBatchWithHiddenStates(tokenBatch, validLengths, training);
            return logits;
        }

        /// <summary>
        /// Versión por batch de <see cref="CalculateCausalLoss"/>: mismo esquema (loss y gradiente
        /// solo desde <c>lossStartIndices[b]</c>, shift de un token, cross-entropy + softmax), pero
        /// para todo un batch en un solo forward/backward vectorizado por lote en vez de un loop de
        /// llamadas individuales. Ver <c>FASE_4_4_CAMBIOS.md</c> para la decisión de diseño sobre
        /// cómo se agregan los gradientes entre ejemplos del batch (suma sin promediar aquí — el
        /// promedio real es responsabilidad de <see cref="AverageGradients"/>, llamado por el
        /// trainer después, exactamente igual que en el esquema por-ejemplo anterior) y sobre el
        /// significado del valor de loss devuelto (promedio por POSICIÓN predicha en todo el batch,
        /// no promedio de promedios por ejemplo).
        /// </summary>
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
                    for (int j = 0; j < _vocabSize; j++)
                    {
                        logitsAtPosition[j] = logits[b, position, j];
                    }

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
            _embedding.UpdateWeights(learningRate);

            foreach (var block in _blocks)
            {
                block.UpdateWeights(learningRate);
            }

            _finalNorm.UpdateWeights(learningRate);

            _outputBiasOptimizer.Update(_outputBias, _outputBiasGradients, learningRate);
            Array.Clear(_outputBiasGradients, 0, _outputBiasGradients.Length);
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