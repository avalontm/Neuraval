using System;
using System.Threading.Tasks;
using Neuraval.Core.Utils;
using Neuraval.Cuda;

namespace Neuraval.Core.Models
{
    public class MultiHeadAttention
    {
        private readonly int _embeddingDim;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly Random _random;

        private float[,] _queryWeights;
        private float[,] _keyWeights;
        private float[,] _valueWeights;
        private float[,] _outputWeights;

        private float[,] _queryGradients;
        private float[,] _keyGradients;
        private float[,] _valueGradients;
        private float[,] _outputGradients;

        private float[,] _accumulatedQueryGradients;
        private float[,] _accumulatedKeyGradients;
        private float[,] _accumulatedValueGradients;
        private float[,] _accumulatedOutputGradients;

        private AdamMatrixOptimizer _queryOptimizer = null!;
        private AdamMatrixOptimizer _keyOptimizer = null!;
        private AdamMatrixOptimizer _valueOptimizer = null!;
        private AdamMatrixOptimizer _outputOptimizer = null!;

        private CudaWeightCache _queryWeightsCache = null!;
        private CudaWeightCache _keyWeightsCache = null!;
        private CudaWeightCache _valueWeightsCache = null!;
        private CudaWeightCache _outputWeightsCache = null!;

        private float[,] _lastInput;
        private float[,,] _lastAttentionWeights;
        private float[,] _lastQueries;
        private float[,] _lastKeys;
        private float[,] _lastValues;
        private float[,] _lastConcatOutput;

        private float[][,]? _scoresBufferPerHead;

        private float[,] GetScoresBuffer(int headIndex, int seqLen)
        {
            if (_scoresBufferPerHead == null || _scoresBufferPerHead.Length != _numHeads)
            {
                _scoresBufferPerHead = new float[_numHeads][,];
            }

            var buffer = _scoresBufferPerHead[headIndex];
            if (buffer == null || buffer.GetLength(0) != seqLen || buffer.GetLength(1) != seqLen)
            {
                buffer = new float[seqLen, seqLen];
                _scoresBufferPerHead[headIndex] = buffer;
            }

            return buffer;
        }

        private float[,,]? _lastInputBatch;
        private float[,,,]? _lastAttentionWeightsBatch;
        private float[,,]? _lastQueriesBatch;
        private float[,,]? _lastKeysBatch;
        private float[,,]? _lastValuesBatch;
        private float[,,]? _lastConcatOutputBatch;

        public int EmbeddingDim => _embeddingDim;
        public int NumHeads => _numHeads;

        public MultiHeadAttention(int embeddingDim, int numHeads, int seed = 42)
        {
            if (embeddingDim % numHeads != 0)
            {
                throw new ArgumentException("Embedding dimension must be divisible by number of heads");
            }

            _embeddingDim = embeddingDim;
            _numHeads = numHeads;
            _headDim = embeddingDim / numHeads;
            _random = new Random(seed);

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            float limit = MathF.Sqrt(6.0f / (_embeddingDim + _embeddingDim));

            _queryWeights = InitializeMatrix(_embeddingDim, _embeddingDim, limit);
            _keyWeights = InitializeMatrix(_embeddingDim, _embeddingDim, limit);
            _valueWeights = InitializeMatrix(_embeddingDim, _embeddingDim, limit);
            _outputWeights = InitializeMatrix(_embeddingDim, _embeddingDim, limit);

            _queryGradients = new float[_embeddingDim, _embeddingDim];
            _keyGradients = new float[_embeddingDim, _embeddingDim];
            _valueGradients = new float[_embeddingDim, _embeddingDim];
            _outputGradients = new float[_embeddingDim, _embeddingDim];

            _accumulatedQueryGradients = new float[_embeddingDim, _embeddingDim];
            _accumulatedKeyGradients = new float[_embeddingDim, _embeddingDim];
            _accumulatedValueGradients = new float[_embeddingDim, _embeddingDim];
            _accumulatedOutputGradients = new float[_embeddingDim, _embeddingDim];

            _queryOptimizer = new AdamMatrixOptimizer(_embeddingDim, _embeddingDim);
            _keyOptimizer = new AdamMatrixOptimizer(_embeddingDim, _embeddingDim);
            _valueOptimizer = new AdamMatrixOptimizer(_embeddingDim, _embeddingDim);
            _outputOptimizer = new AdamMatrixOptimizer(_embeddingDim, _embeddingDim);

            _queryWeightsCache = new CudaWeightCache(_embeddingDim, _embeddingDim);
            _keyWeightsCache = new CudaWeightCache(_embeddingDim, _embeddingDim);
            _valueWeightsCache = new CudaWeightCache(_embeddingDim, _embeddingDim);
            _outputWeightsCache = new CudaWeightCache(_embeddingDim, _embeddingDim);
        }

        private float[,] InitializeMatrix(int rows, int cols, float limit)
        {
            var matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = (_random.NextSingle() * 2 - 1) * limit;
                }
            }
            return matrix;
        }

        public void ZeroGradients()
        {
            Matematicas.ParallelClearMatrix(_accumulatedQueryGradients);
            Matematicas.ParallelClearMatrix(_accumulatedKeyGradients);
            Matematicas.ParallelClearMatrix(_accumulatedValueGradients);
            Matematicas.ParallelClearMatrix(_accumulatedOutputGradients);
        }

        public void AverageGradients(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive");

            float scale = 1.0f / batchSize;

            Parallel.For(0, _embeddingDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    _queryGradients[i, j] = _accumulatedQueryGradients[i, j] * scale;
                    _keyGradients[i, j] = _accumulatedKeyGradients[i, j] * scale;
                    _valueGradients[i, j] = _accumulatedValueGradients[i, j] * scale;
                    _outputGradients[i, j] = _accumulatedOutputGradients[i, j] * scale;
                }
            });
        }

        public void ClipGradients(float maxNorm)
        {
            float totalNorm = 0;
            object lockObj = new object();

            Parallel.For(0, _embeddingDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, () => 0.0f, (i, loop, partial) =>
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    partial += _queryGradients[i, j] * _queryGradients[i, j];
                    partial += _keyGradients[i, j] * _keyGradients[i, j];
                    partial += _valueGradients[i, j] * _valueGradients[i, j];
                    partial += _outputGradients[i, j] * _outputGradients[i, j];
                }
                return partial;
            }, partial =>
            {
                lock (lockObj)
                {
                    totalNorm += partial;
                }
            });

            totalNorm = MathF.Sqrt(totalNorm);

            if (totalNorm > maxNorm)
            {
                float scale = maxNorm / (totalNorm + 1e-10f);

                Parallel.For(0, _embeddingDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
                {
                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        _queryGradients[i, j] *= scale;
                        _keyGradients[i, j] *= scale;
                        _valueGradients[i, j] *= scale;
                        _outputGradients[i, j] *= scale;
                    }
                });
            }
        }

        public float[,] Forward(float[,] input, float[,]? mask = null)
        {
            int seqLen = input.GetLength(0);
            int embDim = input.GetLength(1);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input embedding dimension {embDim} does not match expected {_embeddingDim}");
            }

            _lastInput = (float[,])input.Clone();

            var queries = Matematicas.MatrixMultiplyAutoCached(input, _queryWeights, _queryWeightsCache);
            var keys = Matematicas.MatrixMultiplyAutoCached(input, _keyWeights, _keyWeightsCache);
            var values = Matematicas.MatrixMultiplyAutoCached(input, _valueWeights, _valueWeightsCache);

            _lastQueries = queries;
            _lastKeys = keys;
            _lastValues = values;

            var output = new float[seqLen, _embeddingDim];
            _lastAttentionWeights = new float[_numHeads, seqLen, seqLen];

            var headOutputs = new float[_numHeads][,];

            Parallel.For(0, _numHeads, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, head =>
            {
                int startIdx = head * _headDim;
                int endIdx = startIdx + _headDim;

                var headQueries = Matematicas.SequentialExtractColumns(queries, startIdx, endIdx);
                var headKeys = Matematicas.SequentialExtractColumns(keys, startIdx, endIdx);
                var headValues = Matematicas.SequentialExtractColumns(values, startIdx, endIdx);

                headOutputs[head] = ScaledDotProductAttention(
                    headQueries, headKeys, headValues, mask, head);
            });

            for (int head = 0; head < _numHeads; head++)
            {
                int startIdx = head * _headDim;
                Matematicas.SetColumns(output, headOutputs[head], startIdx);
            }

            _lastConcatOutput = output;

            var finalOutput = Matematicas.MatrixMultiplyAutoCached(output, _outputWeights, _outputWeightsCache);
            return finalOutput;
        }

        private float[,] ScaledDotProductAttention(
            float[,] queries, float[,] keys, float[,] values,
            float[,]? mask, int headIndex)
        {
            int seqLen = queries.GetLength(0);
            float scale = MathF.Sqrt(_headDim);

            var scores = GetScoresBuffer(headIndex, seqLen);
            var computedScores = Matematicas.MatrixMultiplyTransposeBAuto(queries, keys, 1.0f / scale);

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    scores[i, j] = computedScores[i, j];
                }
            }

            if (mask != null)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (mask[i, j] == 0)
                        {
                            scores[i, j] = float.NegativeInfinity;
                        }
                    }
                }
            }

            var attentionWeights = Matematicas.SoftmaxRowsAuto(scores);

            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    _lastAttentionWeights[headIndex, i, j] = attentionWeights[i, j];
                }
            }

            var output = Matematicas.MatrixMultiplyAuto(attentionWeights, values);

            return output;
        }

        public float[,] Backward(float[,] gradOutput, float learningRate)
        {
            if (_lastInput == null || _lastAttentionWeights == null || _lastQueries == null ||
                _lastKeys == null || _lastValues == null || _lastConcatOutput == null)
            {
                throw new InvalidOperationException("Forward must be called before Backward");
            }

            int seqLen = gradOutput.GetLength(0);
            float scale = MathF.Sqrt(_headDim);

            // FinalOutput = ConcatOutput @ Wo
            var gradConcatOutput = Matematicas.MatrixMultiplyTransposeBAutoCached(gradOutput, _outputWeights, _outputWeightsCache);
            var outputWeightsGrad = Matematicas.MatrixMultiplyTransposeAAuto(_lastConcatOutput, gradOutput);
            Matematicas.ParallelMatrixAddInPlace(_accumulatedOutputGradients, outputWeightsGrad);

            var gradQueryHeads = new float[_numHeads][,];
            var gradKeyHeads = new float[_numHeads][,];
            var gradValueHeads = new float[_numHeads][,];

            Parallel.For(0, _numHeads, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, head =>
            {
                int startIdx = head * _headDim;
                int endIdx = startIdx + _headDim;

                var headQueries = Matematicas.SequentialExtractColumns(_lastQueries, startIdx, endIdx);
                var headKeys = Matematicas.SequentialExtractColumns(_lastKeys, startIdx, endIdx);
                var headValues = Matematicas.SequentialExtractColumns(_lastValues, startIdx, endIdx);
                var headGradOutput = Matematicas.SequentialExtractColumns(gradConcatOutput, startIdx, endIdx);

                var attentionWeights = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        attentionWeights[i, j] = _lastAttentionWeights[head, i, j];
                    }
                }

                var gradValueHead = Matematicas.SequentialMatrixMultiply(Matematicas.SequentialTranspose(attentionWeights), headGradOutput);
                var gradAttentionWeights = Matematicas.SequentialMatrixMultiply(headGradOutput, Matematicas.SequentialTranspose(headValues));

                var gradScores = new float[seqLen, seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    float dot = 0;
                    for (int k = 0; k < seqLen; k++)
                    {
                        dot += attentionWeights[i, k] * gradAttentionWeights[i, k];
                    }
                    for (int j = 0; j < seqLen; j++)
                    {
                        gradScores[i, j] = attentionWeights[i, j] * (gradAttentionWeights[i, j] - dot);
                    }
                }

                var gradRawDot = Matematicas.SequentialMatrixScale(gradScores, 1.0f / scale);

                gradQueryHeads[head] = Matematicas.SequentialMatrixMultiply(gradRawDot, headKeys);
                gradKeyHeads[head] = Matematicas.SequentialMatrixMultiply(Matematicas.SequentialTranspose(gradRawDot), headQueries);
                gradValueHeads[head] = gradValueHead;
            });

            var gradQueries = new float[seqLen, _embeddingDim];
            var gradKeys = new float[seqLen, _embeddingDim];
            var gradValues = new float[seqLen, _embeddingDim];

            for (int head = 0; head < _numHeads; head++)
            {
                int startIdx = head * _headDim;
                Matematicas.SetColumns(gradQueries, gradQueryHeads[head], startIdx);
                Matematicas.SetColumns(gradKeys, gradKeyHeads[head], startIdx);
                Matematicas.SetColumns(gradValues, gradValueHeads[head], startIdx);
            }

            var queryWeightsGrad = Matematicas.MatrixMultiplyTransposeAAuto(_lastInput, gradQueries);
            var keyWeightsGrad = Matematicas.MatrixMultiplyTransposeAAuto(_lastInput, gradKeys);
            var valueWeightsGrad = Matematicas.MatrixMultiplyTransposeAAuto(_lastInput, gradValues);

            Matematicas.ParallelMatrixAddInPlace(_accumulatedQueryGradients, queryWeightsGrad);
            Matematicas.ParallelMatrixAddInPlace(_accumulatedKeyGradients, keyWeightsGrad);
            Matematicas.ParallelMatrixAddInPlace(_accumulatedValueGradients, valueWeightsGrad);

            var gradInputFromQuery = Matematicas.MatrixMultiplyTransposeBAutoCached(gradQueries, _queryWeights, _queryWeightsCache);
            var gradInputFromKey = Matematicas.MatrixMultiplyTransposeBAutoCached(gradKeys, _keyWeights, _keyWeightsCache);
            var gradInputFromValue = Matematicas.MatrixMultiplyTransposeBAutoCached(gradValues, _valueWeights, _valueWeightsCache);

            var gradInput = Matematicas.ParallelMatrixAdd(gradInputFromQuery, gradInputFromKey);
            Matematicas.ParallelMatrixAddInPlace(gradInput, gradInputFromValue);

            return gradInput;
        }

        public float[,,] ForwardBatch(float[,,] inputBatch, float[,]? mask = null)
        {
            int batchSize = inputBatch.GetLength(0);
            int seqLen = inputBatch.GetLength(1);
            int embDim = inputBatch.GetLength(2);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input embedding dimension {embDim} does not match expected {_embeddingDim}");
            }

            var output = new float[batchSize, seqLen, _embeddingDim];
            var inputBatchCache = new float[batchSize, seqLen, _embeddingDim];
            var attentionWeightsBatchCache = new float[batchSize, _numHeads, seqLen, seqLen];
            var queriesBatchCache = new float[batchSize, seqLen, _embeddingDim];
            var keysBatchCache = new float[batchSize, seqLen, _embeddingDim];
            var valuesBatchCache = new float[batchSize, seqLen, _embeddingDim];
            var concatOutputBatchCache = new float[batchSize, seqLen, _embeddingDim];

            for (int b = 0; b < batchSize; b++)
            {
                var itemInput = Matematicas.GetBatchSlice(inputBatch, b);
                var itemOutput = Forward(itemInput, mask);

                Matematicas.SetBatchSlice(output, b, itemOutput);
                Matematicas.SetBatchSlice(inputBatchCache, b, _lastInput);
                Matematicas.SetBatchSlice(queriesBatchCache, b, _lastQueries);
                Matematicas.SetBatchSlice(keysBatchCache, b, _lastKeys);
                Matematicas.SetBatchSlice(valuesBatchCache, b, _lastValues);
                Matematicas.SetBatchSlice(concatOutputBatchCache, b, _lastConcatOutput);

                for (int head = 0; head < _numHeads; head++)
                {
                    for (int i = 0; i < seqLen; i++)
                    {
                        for (int j = 0; j < seqLen; j++)
                        {
                            attentionWeightsBatchCache[b, head, i, j] = _lastAttentionWeights[head, i, j];
                        }
                    }
                }
            }

            _lastInputBatch = inputBatchCache;
            _lastAttentionWeightsBatch = attentionWeightsBatchCache;
            _lastQueriesBatch = queriesBatchCache;
            _lastKeysBatch = keysBatchCache;
            _lastValuesBatch = valuesBatchCache;
            _lastConcatOutputBatch = concatOutputBatchCache;

            return output;
        }

        public float[,,] ForwardBatch(float[,,] inputBatch, float[,,]? maskBatch)
        {
            int batchSize = inputBatch.GetLength(0);
            int seqLen = inputBatch.GetLength(1);
            int embDim = inputBatch.GetLength(2);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input embedding dimension {embDim} does not match expected {_embeddingDim}");
            }

            if (maskBatch != null && maskBatch.GetLength(0) != batchSize)
            {
                throw new ArgumentException("maskBatch debe tener el mismo tama�o de batch que inputBatch");
            }

            var output = new float[batchSize, seqLen, _embeddingDim];
            var inputBatchCache = new float[batchSize, seqLen, _embeddingDim];
            var attentionWeightsBatchCache = new float[batchSize, _numHeads, seqLen, seqLen];
            var queriesBatchCache = new float[batchSize, seqLen, _embeddingDim];
            var keysBatchCache = new float[batchSize, seqLen, _embeddingDim];
            var valuesBatchCache = new float[batchSize, seqLen, _embeddingDim];
            var concatOutputBatchCache = new float[batchSize, seqLen, _embeddingDim];

            for (int b = 0; b < batchSize; b++)
            {
                var itemInput = Matematicas.GetBatchSlice(inputBatch, b);
                var itemMask = maskBatch != null ? Matematicas.GetBatchSlice(maskBatch, b) : null;
                var itemOutput = Forward(itemInput, itemMask);

                Matematicas.SetBatchSlice(output, b, itemOutput);
                Matematicas.SetBatchSlice(inputBatchCache, b, _lastInput);
                Matematicas.SetBatchSlice(queriesBatchCache, b, _lastQueries);
                Matematicas.SetBatchSlice(keysBatchCache, b, _lastKeys);
                Matematicas.SetBatchSlice(valuesBatchCache, b, _lastValues);
                Matematicas.SetBatchSlice(concatOutputBatchCache, b, _lastConcatOutput);

                for (int head = 0; head < _numHeads; head++)
                {
                    for (int i = 0; i < seqLen; i++)
                    {
                        for (int j = 0; j < seqLen; j++)
                        {
                            attentionWeightsBatchCache[b, head, i, j] = _lastAttentionWeights[head, i, j];
                        }
                    }
                }
            }

            _lastInputBatch = inputBatchCache;
            _lastAttentionWeightsBatch = attentionWeightsBatchCache;
            _lastQueriesBatch = queriesBatchCache;
            _lastKeysBatch = keysBatchCache;
            _lastValuesBatch = valuesBatchCache;
            _lastConcatOutputBatch = concatOutputBatchCache;

            return output;
        }

        public float[,,] BackwardBatch(float[,,] gradOutputBatch, float learningRate)
        {
            if (_lastInputBatch == null || _lastAttentionWeightsBatch == null || _lastQueriesBatch == null ||
                _lastKeysBatch == null || _lastValuesBatch == null || _lastConcatOutputBatch == null)
            {
                throw new InvalidOperationException("ForwardBatch must be called before BackwardBatch");
            }

            int batchSize = gradOutputBatch.GetLength(0);
            int seqLen = gradOutputBatch.GetLength(1);

            var gradInput = new float[batchSize, seqLen, _embeddingDim];

            for (int b = 0; b < batchSize; b++)
            {
                _lastInput = Matematicas.GetBatchSlice(_lastInputBatch, b);
                _lastQueries = Matematicas.GetBatchSlice(_lastQueriesBatch, b);
                _lastKeys = Matematicas.GetBatchSlice(_lastKeysBatch, b);
                _lastValues = Matematicas.GetBatchSlice(_lastValuesBatch, b);
                _lastConcatOutput = Matematicas.GetBatchSlice(_lastConcatOutputBatch, b);

                var attentionWeightsItem = new float[_numHeads, seqLen, seqLen];
                for (int head = 0; head < _numHeads; head++)
                {
                    for (int i = 0; i < seqLen; i++)
                    {
                        for (int j = 0; j < seqLen; j++)
                        {
                            attentionWeightsItem[head, i, j] = _lastAttentionWeightsBatch[b, head, i, j];
                        }
                    }
                }
                _lastAttentionWeights = attentionWeightsItem;

                var itemGradOutput = Matematicas.GetBatchSlice(gradOutputBatch, b);
                var itemGradInput = Backward(itemGradOutput, learningRate);

                Matematicas.SetBatchSlice(gradInput, b, itemGradInput);
            }

            return gradInput;
        }

        public void UpdateWeights(float learningRate)
        {
            _queryOptimizer.Update(_queryWeights, _queryGradients, learningRate);
            _keyOptimizer.Update(_keyWeights, _keyGradients, learningRate);
            _valueOptimizer.Update(_valueWeights, _valueGradients, learningRate);
            _outputOptimizer.Update(_outputWeights, _outputGradients, learningRate);

            _queryWeightsCache.Invalidate();
            _keyWeightsCache.Invalidate();
            _valueWeightsCache.Invalidate();
            _outputWeightsCache.Invalidate();

            Matematicas.ParallelClearMatrix(_queryGradients);
            Matematicas.ParallelClearMatrix(_keyGradients);
            Matematicas.ParallelClearMatrix(_valueGradients);
            Matematicas.ParallelClearMatrix(_outputGradients);
        }

        public void ResetGradients()
        {
            Matematicas.ParallelClearMatrix(_queryGradients);
            Matematicas.ParallelClearMatrix(_keyGradients);
            Matematicas.ParallelClearMatrix(_valueGradients);
            Matematicas.ParallelClearMatrix(_outputGradients);
            Matematicas.ParallelClearMatrix(_accumulatedQueryGradients);
            Matematicas.ParallelClearMatrix(_accumulatedKeyGradients);
            Matematicas.ParallelClearMatrix(_accumulatedValueGradients);
            Matematicas.ParallelClearMatrix(_accumulatedOutputGradients);
        }

        public MultiHeadAttentionState SaveState()
        {
            return new MultiHeadAttentionState
            {
                EmbeddingDim = _embeddingDim,
                NumHeads = _numHeads,
                QueryWeights = FlattenMatrix(_queryWeights),
                KeyWeights = FlattenMatrix(_keyWeights),
                ValueWeights = FlattenMatrix(_valueWeights),
                OutputWeights = FlattenMatrix(_outputWeights),
                QueryOptimizerState = _queryOptimizer.SaveState(),
                KeyOptimizerState = _keyOptimizer.SaveState(),
                ValueOptimizerState = _valueOptimizer.SaveState(),
                OutputOptimizerState = _outputOptimizer.SaveState()
            };
        }

        private float[] FlattenMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows * cols];
            int index = 0;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[index++] = matrix[i, j];
                }
            }

            return result;
        }

        public static MultiHeadAttention LoadState(MultiHeadAttentionState state)
        {
            var attention = new MultiHeadAttention(state.EmbeddingDim, state.NumHeads);

            attention._queryWeights = UnflattenMatrix(state.QueryWeights, state.EmbeddingDim, state.EmbeddingDim);
            attention._keyWeights = UnflattenMatrix(state.KeyWeights, state.EmbeddingDim, state.EmbeddingDim);
            attention._valueWeights = UnflattenMatrix(state.ValueWeights, state.EmbeddingDim, state.EmbeddingDim);
            attention._outputWeights = UnflattenMatrix(state.OutputWeights, state.EmbeddingDim, state.EmbeddingDim);

            if (state.QueryOptimizerState != null) attention._queryOptimizer.LoadStateInto(state.QueryOptimizerState);
            if (state.KeyOptimizerState != null) attention._keyOptimizer.LoadStateInto(state.KeyOptimizerState);
            if (state.ValueOptimizerState != null) attention._valueOptimizer.LoadStateInto(state.ValueOptimizerState);
            if (state.OutputOptimizerState != null) attention._outputOptimizer.LoadStateInto(state.OutputOptimizerState);

            attention._queryWeightsCache.Invalidate();
            attention._keyWeightsCache.Invalidate();
            attention._valueWeightsCache.Invalidate();
            attention._outputWeightsCache.Invalidate();

            return attention;
        }

        private static float[,] UnflattenMatrix(float[] array, int rows, int cols)
        {
            var matrix = new float[rows, cols];
            int index = 0;

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = array[index++];
                }
            }

            return matrix;
        }
    }

    public class MultiHeadAttentionState
    {
        public int EmbeddingDim { get; set; }
        public int NumHeads { get; set; }
        public float[] QueryWeights { get; set; }
        public float[] KeyWeights { get; set; }
        public float[] ValueWeights { get; set; }
        public float[] OutputWeights { get; set; }
        public AdamMatrixOptimizerState? QueryOptimizerState { get; set; }
        public AdamMatrixOptimizerState? KeyOptimizerState { get; set; }
        public AdamMatrixOptimizerState? ValueOptimizerState { get; set; }
        public AdamMatrixOptimizerState? OutputOptimizerState { get; set; }

        public MultiHeadAttentionState()
        {
            QueryWeights = Array.Empty<float>();
            KeyWeights = Array.Empty<float>();
            ValueWeights = Array.Empty<float>();
            OutputWeights = Array.Empty<float>();
        }
    }
}