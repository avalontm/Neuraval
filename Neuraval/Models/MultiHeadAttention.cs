using System;
using System.Linq;
using System.Threading.Tasks;
using Neuraval.Core.Quantization;
using Neuraval.Core.Utils;
using Neuraval.Cuda;
using Neuraval.Tensor;

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

        private Neuraval.Tensor.Tensor _queryGradients;
        private Neuraval.Tensor.Tensor _keyGradients;
        private Neuraval.Tensor.Tensor _valueGradients;
        private Neuraval.Tensor.Tensor _outputGradients;

        private Neuraval.Tensor.Tensor _accumulatedQueryGradients;
        private Neuraval.Tensor.Tensor _accumulatedKeyGradients;
        private Neuraval.Tensor.Tensor _accumulatedValueGradients;
        private Neuraval.Tensor.Tensor _accumulatedOutputGradients;

        private AdamMatrixOptimizer _queryOptimizer = null!;
        private AdamMatrixOptimizer _keyOptimizer = null!;
        private AdamMatrixOptimizer _valueOptimizer = null!;
        private AdamMatrixOptimizer _outputOptimizer = null!;

        private CudaWeightCache _queryWeightsCache = null!;
        private CudaWeightCache _keyWeightsCache = null!;
        private CudaWeightCache _valueWeightsCache = null!;
        private CudaWeightCache _outputWeightsCache = null!;

        private CudaWeightCacheFp16 _queryWeightsCacheFp16 = null!;
        private CudaWeightCacheFp16 _keyWeightsCacheFp16 = null!;
        private CudaWeightCacheFp16 _valueWeightsCacheFp16 = null!;
        private CudaWeightCacheFp16 _outputWeightsCacheFp16 = null!;

        private Int8WeightCache _queryWeightsCacheInt8 = null!;
        private Int8WeightCache _keyWeightsCacheInt8 = null!;
        private Int8WeightCache _valueWeightsCacheInt8 = null!;
        private Int8WeightCache _outputWeightsCacheInt8 = null!;

        private LoraProjection? _queryLora;
        private LoraProjection? _keyLora;
        private LoraProjection? _valueLora;
        private LoraProjection? _outputLora;
        private bool _freezeBaseWeights;

        private float[,] _lastInput;
        private float[,,] _lastAttentionWeights;
        private float[,] _lastQueries;
        private float[,] _lastKeys;
        private float[,] _lastValues;
        private float[,] _lastConcatOutput;

        private float[,,]? _lastInputBatch;
        private float[,,,]? _lastAttentionWeightsBatch;
        private float[,,]? _lastQueriesBatch;
        private float[,,]? _lastKeysBatch;
        private float[,,]? _lastValuesBatch;
        private float[,,]? _lastConcatOutputBatch;

        public int EmbeddingDim => _embeddingDim;
        public int NumHeads => _numHeads;
        public bool HasLora => _queryLora != null;
        public bool FreezeBaseWeightsEnabled => _freezeBaseWeights;

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

            _queryGradients = new Neuraval.Tensor.Tensor(new[] { _embeddingDim, _embeddingDim });
            _keyGradients = new Neuraval.Tensor.Tensor(new[] { _embeddingDim, _embeddingDim });
            _valueGradients = new Neuraval.Tensor.Tensor(new[] { _embeddingDim, _embeddingDim });
            _outputGradients = new Neuraval.Tensor.Tensor(new[] { _embeddingDim, _embeddingDim });

            _accumulatedQueryGradients = new Neuraval.Tensor.Tensor(new[] { _embeddingDim, _embeddingDim });
            _accumulatedKeyGradients = new Neuraval.Tensor.Tensor(new[] { _embeddingDim, _embeddingDim });
            _accumulatedValueGradients = new Neuraval.Tensor.Tensor(new[] { _embeddingDim, _embeddingDim });
            _accumulatedOutputGradients = new Neuraval.Tensor.Tensor(new[] { _embeddingDim, _embeddingDim });

            _queryOptimizer = new AdamMatrixOptimizer(_embeddingDim, _embeddingDim);
            _keyOptimizer = new AdamMatrixOptimizer(_embeddingDim, _embeddingDim);
            _valueOptimizer = new AdamMatrixOptimizer(_embeddingDim, _embeddingDim);
            _outputOptimizer = new AdamMatrixOptimizer(_embeddingDim, _embeddingDim);

            _queryWeightsCache = new CudaWeightCache(_embeddingDim, _embeddingDim);
            _keyWeightsCache = new CudaWeightCache(_embeddingDim, _embeddingDim);
            _valueWeightsCache = new CudaWeightCache(_embeddingDim, _embeddingDim);
            _outputWeightsCache = new CudaWeightCache(_embeddingDim, _embeddingDim);

            _queryWeightsCacheFp16 = new CudaWeightCacheFp16(_embeddingDim, _embeddingDim);
            _keyWeightsCacheFp16 = new CudaWeightCacheFp16(_embeddingDim, _embeddingDim);
            _valueWeightsCacheFp16 = new CudaWeightCacheFp16(_embeddingDim, _embeddingDim);
            _outputWeightsCacheFp16 = new CudaWeightCacheFp16(_embeddingDim, _embeddingDim);

            _queryWeightsCacheInt8 = new Int8WeightCache(_embeddingDim, _embeddingDim);
            _keyWeightsCacheInt8 = new Int8WeightCache(_embeddingDim, _embeddingDim);
            _valueWeightsCacheInt8 = new Int8WeightCache(_embeddingDim, _embeddingDim);
            _outputWeightsCacheInt8 = new Int8WeightCache(_embeddingDim, _embeddingDim);
        }

        public void EnableLora(int rank, float alpha, int seed = 9001)
        {
            if (_queryLora != null) return;

            _queryLora = new LoraProjection(_embeddingDim, _embeddingDim, rank, alpha, seed);
            _keyLora = new LoraProjection(_embeddingDim, _embeddingDim, rank, alpha, seed + 1);
            _valueLora = new LoraProjection(_embeddingDim, _embeddingDim, rank, alpha, seed + 2);
            _outputLora = new LoraProjection(_embeddingDim, _embeddingDim, rank, alpha, seed + 3);
        }

        public void SetFreezeBaseWeights(bool freeze)
        {
            _freezeBaseWeights = freeze;
        }

        public LoraAttentionState? SaveLoraState()
        {
            if (_queryLora == null) return null;

            return new LoraAttentionState
            {
                Query = _queryLora.SaveState(),
                Key = _keyLora!.SaveState(),
                Value = _valueLora!.SaveState(),
                Output = _outputLora!.SaveState(),
                FreezeBaseWeights = _freezeBaseWeights
            };
        }

        public void LoadLoraState(LoraAttentionState state)
        {
            _queryLora = LoraProjection.LoadState(state.Query);
            _keyLora = LoraProjection.LoadState(state.Key);
            _valueLora = LoraProjection.LoadState(state.Value);
            _outputLora = LoraProjection.LoadState(state.Output);
            _freezeBaseWeights = state.FreezeBaseWeights;
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
            TensorOps.Clear(_accumulatedQueryGradients);
            TensorOps.Clear(_accumulatedKeyGradients);
            TensorOps.Clear(_accumulatedValueGradients);
            TensorOps.Clear(_accumulatedOutputGradients);

            _queryLora?.ZeroGradients();
            _keyLora?.ZeroGradients();
            _valueLora?.ZeroGradients();
            _outputLora?.ZeroGradients();
        }

        public void AverageGradients(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive");

            float scale = 1.0f / batchSize;

            _queryGradients = TensorOps.Scale(_accumulatedQueryGradients, scale);
            _keyGradients = TensorOps.Scale(_accumulatedKeyGradients, scale);
            _valueGradients = TensorOps.Scale(_accumulatedValueGradients, scale);
            _outputGradients = TensorOps.Scale(_accumulatedOutputGradients, scale);

            _queryLora?.AverageGradients(batchSize);
            _keyLora?.AverageGradients(batchSize);
            _valueLora?.AverageGradients(batchSize);
            _outputLora?.AverageGradients(batchSize);
        }

        public float SumSquaredGradients()
        {
            float totalSumSquared = 0;

            totalSumSquared += SumSquared(_queryGradients);
            totalSumSquared += SumSquared(_keyGradients);
            totalSumSquared += SumSquared(_valueGradients);
            totalSumSquared += SumSquared(_outputGradients);

            if (_queryLora != null) totalSumSquared += _queryLora.SumSquaredGradients();
            if (_keyLora != null) totalSumSquared += _keyLora.SumSquaredGradients();
            if (_valueLora != null) totalSumSquared += _valueLora.SumSquaredGradients();
            if (_outputLora != null) totalSumSquared += _outputLora.SumSquaredGradients();

            return totalSumSquared;
        }

        private static float SumSquared(Neuraval.Tensor.Tensor tensor)
        {
            return TensorOps.Sum(TensorOps.Multiply(tensor, tensor));
        }

        public void ScaleGradients(float scale)
        {
            _queryGradients = TensorOps.Scale(_queryGradients, scale);
            _keyGradients = TensorOps.Scale(_keyGradients, scale);
            _valueGradients = TensorOps.Scale(_valueGradients, scale);
            _outputGradients = TensorOps.Scale(_outputGradients, scale);

            _queryLora?.ScaleGradients(scale);
            _keyLora?.ScaleGradients(scale);
            _valueLora?.ScaleGradients(scale);
            _outputLora?.ScaleGradients(scale);
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

            var device = TensorDeviceSelector.Current;

            try
            {
                var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(input, device);

                var queryTensor = TensorOps.MatMulCachedB(inputTensor, _queryWeights, _queryWeightsCache);
                var keyTensor = TensorOps.MatMulCachedB(inputTensor, _keyWeights, _keyWeightsCache);
                var valueTensor = TensorOps.MatMulCachedB(inputTensor, _valueWeights, _valueWeightsCache);

                if (_queryLora != null) TensorOps.AddInPlace(queryTensor, _queryLora.Forward(inputTensor, device));
                if (_keyLora != null) TensorOps.AddInPlace(keyTensor, _keyLora.Forward(inputTensor, device));
                if (_valueLora != null) TensorOps.AddInPlace(valueTensor, _valueLora.Forward(inputTensor, device));

                _lastQueries = queryTensor.ToArray2D();
                _lastKeys = keyTensor.ToArray2D();
                _lastValues = valueTensor.ToArray2D();

                var outputTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _embeddingDim }, device);
                _lastAttentionWeights = new float[_numHeads, seqLen, seqLen];

                var headOutputs = new Neuraval.Tensor.Tensor[_numHeads];

                Parallel.For(0, _numHeads, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, head =>
                {
                    int startIdx = head * _headDim;
                    int endIdx = startIdx + _headDim;

                    var headQueries = TensorOps.SliceColumns(queryTensor, startIdx, endIdx);
                    var headKeys = TensorOps.SliceColumns(keyTensor, startIdx, endIdx);
                    var headValues = TensorOps.SliceColumns(valueTensor, startIdx, endIdx);

                    headOutputs[head] = ScaledDotProductAttention(headQueries, headKeys, headValues, mask, head, device);
                });

                for (int head = 0; head < _numHeads; head++)
                {
                    int startIdx = head * _headDim;
                    TensorOps.SetColumns(outputTensor, headOutputs[head], startIdx);
                }

                var output = outputTensor.ToArray2D();
                _lastConcatOutput = output;

                var finalOutputTensor = TensorOps.MatMulCachedB(outputTensor, _outputWeights, _outputWeightsCache);

                if (_outputLora != null) TensorOps.AddInPlace(finalOutputTensor, _outputLora.Forward(outputTensor, device));

                return finalOutputTensor.ToArray2D();
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return Forward(input, mask);
            }
            catch (AggregateException ex) when (device == DeviceType.Cuda && ex.InnerExceptions.Any(inner => inner is CudaException))
            {
                TensorDeviceSelector.ReportFailure();
                return Forward(input, mask);
            }
        }

        private Neuraval.Tensor.Tensor ScaledDotProductAttention(
            Neuraval.Tensor.Tensor queries, Neuraval.Tensor.Tensor keys, Neuraval.Tensor.Tensor values,
            float[,]? mask, int headIndex, DeviceType device)
        {
            int seqLen = queries.Shape[0];
            float scale = MathF.Sqrt(_headDim);

            var scoresTensor = TensorOps.MatMulTransposeB(queries, keys, 1.0f / scale);

            if (mask != null)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        if (mask[i, j] == 0)
                        {
                            scoresTensor.Buffer[i * seqLen + j] = float.NegativeInfinity;
                        }
                    }
                }
            }

            var attentionWeightsTensor = TensorOps.SoftmaxRows(scoresTensor);

            Buffer.BlockCopy(attentionWeightsTensor.Buffer, 0, _lastAttentionWeights, headIndex * seqLen * seqLen * sizeof(float), seqLen * seqLen * sizeof(float));

            return TensorOps.MatMul(attentionWeightsTensor, values);
        }

        private static Neuraval.Tensor.Tensor MatMulCachedBInference(
            Neuraval.Tensor.Tensor input, float[,] weights,
            CudaWeightCache fp32Cache, CudaWeightCacheFp16 fp16Cache, Int8WeightCache int8Cache, DeviceType device)
        {
            if (device == DeviceType.Cuda && MixedPrecisionSettings.EnableFp16Inference)
            {
                int mFp16 = input.Shape[0];
                int kFp16 = input.Shape[1];
                int nFp16 = weights.GetLength(1);

                var resultFp16 = CudaMath.MatrixMultiplyCachedBHalf(input.Buffer, weights, fp16Cache, mFp16, kFp16, nFp16);
                return new Neuraval.Tensor.Tensor(resultFp16, new[] { mFp16, nFp16 }, device, input.DType);
            }

            if (device == DeviceType.Cpu && Int8InferenceSettings.EnableInt8Cpu)
            {
                return Int8MatMul.MatMulCachedB(input, weights, int8Cache);
            }

            return TensorOps.MatMulCachedB(input, weights, fp32Cache);
        }

        public float[,] ForwardInference(float[,] input, bool causal)
        {
            int seqLen = input.GetLength(0);
            int embDim = input.GetLength(1);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input embedding dimension {embDim} does not match expected {_embeddingDim}");
            }

            var device = TensorDeviceSelector.Current;

            try
            {
                var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(input, device);

                var queriesTensor = MatMulCachedBInference(inputTensor, _queryWeights, _queryWeightsCache, _queryWeightsCacheFp16, _queryWeightsCacheInt8, device);
                var keysTensor = MatMulCachedBInference(inputTensor, _keyWeights, _keyWeightsCache, _keyWeightsCacheFp16, _keyWeightsCacheInt8, device);
                var valuesTensor = MatMulCachedBInference(inputTensor, _valueWeights, _valueWeightsCache, _valueWeightsCacheFp16, _valueWeightsCacheInt8, device);

                // Los adaptadores LoRA siempre se aplican en fp32 (son chicos:
                // rank x embeddingDim), sin importar el modo de precisión
                // (fp16/INT8) elegido para los pesos base.
                if (_queryLora != null) TensorOps.AddInPlace(queriesTensor, _queryLora.Forward(inputTensor, device));
                if (_keyLora != null) TensorOps.AddInPlace(keysTensor, _keyLora.Forward(inputTensor, device));
                if (_valueLora != null) TensorOps.AddInPlace(valuesTensor, _valueLora.Forward(inputTensor, device));

                float scaleMultiplier = 1.0f / MathF.Sqrt(_headDim);
                var outputTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _embeddingDim }, device);
                var headOutputs = new Neuraval.Tensor.Tensor[_numHeads];

                Parallel.For(0, _numHeads, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, head =>
                {
                    int startIdx = head * _headDim;
                    int endIdx = startIdx + _headDim;

                    var headQueriesTensor = TensorOps.SliceColumns(queriesTensor, startIdx, endIdx);
                    var headKeysTensor = TensorOps.SliceColumns(keysTensor, startIdx, endIdx);
                    var headValuesTensor = TensorOps.SliceColumns(valuesTensor, startIdx, endIdx);

                    headOutputs[head] = FlashAttentionOps.Attend(headQueriesTensor, headKeysTensor, headValuesTensor, causal, scaleMultiplier);
                });

                for (int head = 0; head < _numHeads; head++)
                {
                    TensorOps.SetColumns(outputTensor, headOutputs[head], head * _headDim);
                }

                return MatMulCachedBInference(outputTensor, _outputWeights, _outputWeightsCache, _outputWeightsCacheFp16, _outputWeightsCacheInt8, device).ToArray2D();
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return ForwardInference(input, causal);
            }
            catch (AggregateException ex) when (device == DeviceType.Cuda && ex.InnerExceptions.Any(inner => inner is CudaException))
            {
                TensorDeviceSelector.ReportFailure();
                return ForwardInference(input, causal);
            }
        }

        public float[,] ForwardIncremental(float[,] newInput, KVCacheLayer cache)
        {
            int newSeqLen = newInput.GetLength(0);
            int embDim = newInput.GetLength(1);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input embedding dimension {embDim} does not match expected {_embeddingDim}");
            }

            var device = TensorDeviceSelector.Current;

            try
            {
                var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(newInput, device);

                var newQueriesTensor = MatMulCachedBInference(inputTensor, _queryWeights, _queryWeightsCache, _queryWeightsCacheFp16, _queryWeightsCacheInt8, device);
                var newKeys = MatMulCachedBInference(inputTensor, _keyWeights, _keyWeightsCache, _keyWeightsCacheFp16, _keyWeightsCacheInt8, device).ToArray2D();
                var newValues = MatMulCachedBInference(inputTensor, _valueWeights, _valueWeightsCache, _valueWeightsCacheFp16, _valueWeightsCacheInt8, device).ToArray2D();

                int pastLength = cache.Length;
                cache.Append(newKeys, newValues);

                var keysTensor = Neuraval.Tensor.Tensor.FromArray2D(cache.GetKeys(), device);
                var valuesTensor = Neuraval.Tensor.Tensor.FromArray2D(cache.GetValues(), device);

                var outputTensor = new Neuraval.Tensor.Tensor(new[] { newSeqLen, _embeddingDim }, device);
                var headOutputs = new Neuraval.Tensor.Tensor[_numHeads];

                Parallel.For(0, _numHeads, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, head =>
                {
                    int startIdx = head * _headDim;
                    int endIdx = startIdx + _headDim;

                    var headQueries = TensorOps.SliceColumns(newQueriesTensor, startIdx, endIdx);
                    var headKeys = TensorOps.SliceColumns(keysTensor, startIdx, endIdx);
                    var headValues = TensorOps.SliceColumns(valuesTensor, startIdx, endIdx);

                    headOutputs[head] = IncrementalScaledDotProductAttention(headQueries, headKeys, headValues, pastLength, device);
                });

                for (int head = 0; head < _numHeads; head++)
                {
                    TensorOps.SetColumns(outputTensor, headOutputs[head], head * _headDim);
                }

                return MatMulCachedBInference(outputTensor, _outputWeights, _outputWeightsCache, _outputWeightsCacheFp16, _outputWeightsCacheInt8, device).ToArray2D();
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return ForwardIncremental(newInput, cache);
            }
            catch (AggregateException ex) when (device == DeviceType.Cuda && ex.InnerExceptions.Any(inner => inner is CudaException))
            {
                TensorDeviceSelector.ReportFailure();
                return ForwardIncremental(newInput, cache);
            }
        }

        private Neuraval.Tensor.Tensor IncrementalScaledDotProductAttention(
            Neuraval.Tensor.Tensor queries, Neuraval.Tensor.Tensor keys, Neuraval.Tensor.Tensor values, int pastLength, DeviceType device)
        {
            float scaleMultiplier = 1.0f / MathF.Sqrt(_headDim);

            return FlashAttentionOps.Attend(queries, keys, values, true, scaleMultiplier, pastLength);
        }

        public float QueryGradientAt(int i, int j) => _queryGradients[i, j];

        public float KeyGradientAt(int i, int j) => _keyGradients[i, j];

        public float ValueGradientAt(int i, int j) => _valueGradients[i, j];

        public float OutputGradientAt(int i, int j) => _outputGradients[i, j];

        public float[,] Backward(float[,] gradOutput, float learningRate)
        {
            if (_lastInput == null || _lastAttentionWeights == null || _lastQueries == null ||
                _lastKeys == null || _lastValues == null || _lastConcatOutput == null)
            {
                throw new InvalidOperationException("Forward must be called before Backward");
            }

            var device = TensorDeviceSelector.Current;

            try
            {
                return BackwardCore(gradOutput, device);
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return Backward(gradOutput, learningRate);
            }
            catch (AggregateException ex) when (device == DeviceType.Cuda && ex.InnerExceptions.Any(inner => inner is CudaException))
            {
                TensorDeviceSelector.ReportFailure();
                return Backward(gradOutput, learningRate);
            }
        }

        private float[,] BackwardCore(float[,] gradOutput, DeviceType device)
        {
            int seqLen = gradOutput.GetLength(0);
            float scale = MathF.Sqrt(_headDim);

            var gradOutputTensor = Neuraval.Tensor.Tensor.FromArray2D(gradOutput, device);
            var concatOutputTensor = Neuraval.Tensor.Tensor.FromArray2D(_lastConcatOutput, device);

            var gradConcatOutputTensor = TensorOps.MatMulTransposeBCachedB(gradOutputTensor, _outputWeights, _outputWeightsCache);
            if (_outputLora != null) TensorOps.AddInPlace(gradConcatOutputTensor, _outputLora.Backward(gradOutputTensor, device));
            var outputWeightsGrad = TensorOps.MatMulTransposeA(concatOutputTensor, gradOutputTensor);

            var queriesTensor = Neuraval.Tensor.Tensor.FromArray2D(_lastQueries, device);
            var keysTensor = Neuraval.Tensor.Tensor.FromArray2D(_lastKeys, device);
            var valuesTensor = Neuraval.Tensor.Tensor.FromArray2D(_lastValues, device);

            var gradQueryHeads = new Neuraval.Tensor.Tensor[_numHeads];
            var gradKeyHeads = new Neuraval.Tensor.Tensor[_numHeads];
            var gradValueHeads = new Neuraval.Tensor.Tensor[_numHeads];

            Parallel.For(0, _numHeads, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, head =>
            {
                int startIdx = head * _headDim;
                int endIdx = startIdx + _headDim;

                var headQueriesTensor = TensorOps.SliceColumns(queriesTensor, startIdx, endIdx);
                var headKeysTensor = TensorOps.SliceColumns(keysTensor, startIdx, endIdx);
                var headValuesTensor = TensorOps.SliceColumns(valuesTensor, startIdx, endIdx);
                var headGradOutputTensor = TensorOps.SliceColumns(gradConcatOutputTensor, startIdx, endIdx);

                // _lastAttentionWeights[head] ya es un bloque contiguo dentro del
                // float[,,]; copiarlo directo al Buffer del tensor evita el paso
                // intermedio por un float[,] + conversión elemento a elemento.
                var attentionWeightsTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, seqLen }, device);
                System.Buffer.BlockCopy(_lastAttentionWeights, head * seqLen * seqLen * sizeof(float),
                    attentionWeightsTensor.Buffer, 0, seqLen * seqLen * sizeof(float));

                var gradValueHead = TensorOps.MatMulTransposeA(attentionWeightsTensor, headGradOutputTensor);
                var gradAttentionWeights = TensorOps.MatMulTransposeB(headGradOutputTensor, headValuesTensor);

                var gradScores = TensorOps.SoftmaxRowsBackward(gradAttentionWeights, attentionWeightsTensor);
                var gradRawDot = TensorOps.Scale(gradScores, 1.0f / scale);

                gradQueryHeads[head] = TensorOps.MatMul(gradRawDot, headKeysTensor);
                gradKeyHeads[head] = TensorOps.MatMulTransposeA(gradRawDot, headQueriesTensor);
                gradValueHeads[head] = gradValueHead;
            });

            var gradQueriesTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _embeddingDim }, device);
            var gradKeysTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _embeddingDim }, device);
            var gradValuesTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _embeddingDim }, device);

            for (int head = 0; head < _numHeads; head++)
            {
                int startIdx = head * _headDim;
                TensorOps.SetColumns(gradQueriesTensor, gradQueryHeads[head], startIdx);
                TensorOps.SetColumns(gradKeysTensor, gradKeyHeads[head], startIdx);
                TensorOps.SetColumns(gradValuesTensor, gradValueHeads[head], startIdx);
            }

            var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(_lastInput, device);

            var queryWeightsGrad = TensorOps.MatMulTransposeA(inputTensor, gradQueriesTensor);
            var keyWeightsGrad = TensorOps.MatMulTransposeA(inputTensor, gradKeysTensor);
            var valueWeightsGrad = TensorOps.MatMulTransposeA(inputTensor, gradValuesTensor);

            var gradInputTensor = TensorOps.MatMulTransposeBCachedB(gradQueriesTensor, _queryWeights, _queryWeightsCache);
            var gradInputFromKey = TensorOps.MatMulTransposeBCachedB(gradKeysTensor, _keyWeights, _keyWeightsCache);
            var gradInputFromValue = TensorOps.MatMulTransposeBCachedB(gradValuesTensor, _valueWeights, _valueWeightsCache);

            TensorOps.AddInPlace(gradInputTensor, gradInputFromKey);
            TensorOps.AddInPlace(gradInputTensor, gradInputFromValue);

            if (_queryLora != null) TensorOps.AddInPlace(gradInputTensor, _queryLora.Backward(gradQueriesTensor, device));
            if (_keyLora != null) TensorOps.AddInPlace(gradInputTensor, _keyLora.Backward(gradKeysTensor, device));
            if (_valueLora != null) TensorOps.AddInPlace(gradInputTensor, _valueLora.Backward(gradValuesTensor, device));

            TensorOps.AddInPlace(_accumulatedOutputGradients, outputWeightsGrad);
            TensorOps.AddInPlace(_accumulatedQueryGradients, queryWeightsGrad);
            TensorOps.AddInPlace(_accumulatedKeyGradients, keyWeightsGrad);
            TensorOps.AddInPlace(_accumulatedValueGradients, valueWeightsGrad);

            return gradInputTensor.ToArray2D();
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

            var inputBatchTensor = Neuraval.Tensor.Tensor.FromArray3D(inputBatch);

            var outputTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var inputBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var queriesBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var keysBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var valuesBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var concatOutputBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var attentionWeightsBatchCache = new float[batchSize, _numHeads, seqLen, seqLen];

            for (int b = 0; b < batchSize; b++)
            {
                var itemInput = TensorOps.GetBatchSlice(inputBatchTensor, b).ToArray2D();
                var itemOutput = Forward(itemInput, mask);

                TensorOps.SetBatchSlice(outputTensor, b, Neuraval.Tensor.Tensor.FromArray2D(itemOutput));
                TensorOps.SetBatchSlice(inputBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastInput));
                TensorOps.SetBatchSlice(queriesBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastQueries));
                TensorOps.SetBatchSlice(keysBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastKeys));
                TensorOps.SetBatchSlice(valuesBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastValues));
                TensorOps.SetBatchSlice(concatOutputBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastConcatOutput));

                int attnSlabSize = _numHeads * seqLen * seqLen;
                System.Buffer.BlockCopy(_lastAttentionWeights, 0, attentionWeightsBatchCache,
                    b * attnSlabSize * sizeof(float), attnSlabSize * sizeof(float));
            }

            _lastInputBatch = inputBatchCacheTensor.ToArray3D();
            _lastAttentionWeightsBatch = attentionWeightsBatchCache;
            _lastQueriesBatch = queriesBatchCacheTensor.ToArray3D();
            _lastKeysBatch = keysBatchCacheTensor.ToArray3D();
            _lastValuesBatch = valuesBatchCacheTensor.ToArray3D();
            _lastConcatOutputBatch = concatOutputBatchCacheTensor.ToArray3D();

            return outputTensor.ToArray3D();
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

            var inputBatchTensor = Neuraval.Tensor.Tensor.FromArray3D(inputBatch);
            var maskBatchTensor = maskBatch != null ? Neuraval.Tensor.Tensor.FromArray3D(maskBatch) : null;

            var outputTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var inputBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var queriesBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var keysBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var valuesBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var concatOutputBatchCacheTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });
            var attentionWeightsBatchCache = new float[batchSize, _numHeads, seqLen, seqLen];

            for (int b = 0; b < batchSize; b++)
            {
                var itemInput = TensorOps.GetBatchSlice(inputBatchTensor, b).ToArray2D();
                var itemMask = maskBatchTensor != null ? TensorOps.GetBatchSlice(maskBatchTensor, b).ToArray2D() : null;
                var itemOutput = Forward(itemInput, itemMask);

                TensorOps.SetBatchSlice(outputTensor, b, Neuraval.Tensor.Tensor.FromArray2D(itemOutput));
                TensorOps.SetBatchSlice(inputBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastInput));
                TensorOps.SetBatchSlice(queriesBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastQueries));
                TensorOps.SetBatchSlice(keysBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastKeys));
                TensorOps.SetBatchSlice(valuesBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastValues));
                TensorOps.SetBatchSlice(concatOutputBatchCacheTensor, b, Neuraval.Tensor.Tensor.FromArray2D(_lastConcatOutput));

                int attnSlabSize = _numHeads * seqLen * seqLen;
                System.Buffer.BlockCopy(_lastAttentionWeights, 0, attentionWeightsBatchCache,
                    b * attnSlabSize * sizeof(float), attnSlabSize * sizeof(float));
            }

            _lastInputBatch = inputBatchCacheTensor.ToArray3D();
            _lastAttentionWeightsBatch = attentionWeightsBatchCache;
            _lastQueriesBatch = queriesBatchCacheTensor.ToArray3D();
            _lastKeysBatch = keysBatchCacheTensor.ToArray3D();
            _lastValuesBatch = valuesBatchCacheTensor.ToArray3D();
            _lastConcatOutputBatch = concatOutputBatchCacheTensor.ToArray3D();

            return outputTensor.ToArray3D();
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

            var lastInputBatchTensor = Neuraval.Tensor.Tensor.FromArray3D(_lastInputBatch);
            var lastQueriesBatchTensor = Neuraval.Tensor.Tensor.FromArray3D(_lastQueriesBatch);
            var lastKeysBatchTensor = Neuraval.Tensor.Tensor.FromArray3D(_lastKeysBatch);
            var lastValuesBatchTensor = Neuraval.Tensor.Tensor.FromArray3D(_lastValuesBatch);
            var lastConcatOutputBatchTensor = Neuraval.Tensor.Tensor.FromArray3D(_lastConcatOutputBatch);
            var gradOutputBatchTensor = Neuraval.Tensor.Tensor.FromArray3D(gradOutputBatch);

            var gradInputTensor = new Neuraval.Tensor.Tensor(new[] { batchSize, seqLen, _embeddingDim });

            for (int b = 0; b < batchSize; b++)
            {
                _lastInput = TensorOps.GetBatchSlice(lastInputBatchTensor, b).ToArray2D();
                _lastQueries = TensorOps.GetBatchSlice(lastQueriesBatchTensor, b).ToArray2D();
                _lastKeys = TensorOps.GetBatchSlice(lastKeysBatchTensor, b).ToArray2D();
                _lastValues = TensorOps.GetBatchSlice(lastValuesBatchTensor, b).ToArray2D();
                _lastConcatOutput = TensorOps.GetBatchSlice(lastConcatOutputBatchTensor, b).ToArray2D();

                var attentionWeightsItem = new float[_numHeads, seqLen, seqLen];
                int attnSlabSize = _numHeads * seqLen * seqLen;
                System.Buffer.BlockCopy(_lastAttentionWeightsBatch, b * attnSlabSize * sizeof(float),
                    attentionWeightsItem, 0, attnSlabSize * sizeof(float));
                _lastAttentionWeights = attentionWeightsItem;

                var itemGradOutput = TensorOps.GetBatchSlice(gradOutputBatchTensor, b).ToArray2D();
                var itemGradInput = Backward(itemGradOutput, learningRate);

                TensorOps.SetBatchSlice(gradInputTensor, b, Neuraval.Tensor.Tensor.FromArray2D(itemGradInput));
            }

            return gradInputTensor.ToArray3D();
        }

        public void UpdateWeights(float learningRate)
        {
            if (!_freezeBaseWeights)
            {
                _queryOptimizer.Update(_queryWeights, _queryGradients.ToArray2D(), learningRate);
                _keyOptimizer.Update(_keyWeights, _keyGradients.ToArray2D(), learningRate);
                _valueOptimizer.Update(_valueWeights, _valueGradients.ToArray2D(), learningRate);
                _outputOptimizer.Update(_outputWeights, _outputGradients.ToArray2D(), learningRate);

                _queryWeightsCache.Invalidate();
                _keyWeightsCache.Invalidate();
                _valueWeightsCache.Invalidate();
                _outputWeightsCache.Invalidate();

                _queryWeightsCacheFp16.Invalidate();
                _keyWeightsCacheFp16.Invalidate();
                _valueWeightsCacheFp16.Invalidate();
                _outputWeightsCacheFp16.Invalidate();

                _queryWeightsCacheInt8.Invalidate();
                _keyWeightsCacheInt8.Invalidate();
                _valueWeightsCacheInt8.Invalidate();
                _outputWeightsCacheInt8.Invalidate();
            }

            _queryLora?.UpdateWeights(learningRate);
            _keyLora?.UpdateWeights(learningRate);
            _valueLora?.UpdateWeights(learningRate);
            _outputLora?.UpdateWeights(learningRate);

            TensorOps.Clear(_queryGradients);
            TensorOps.Clear(_keyGradients);
            TensorOps.Clear(_valueGradients);
            TensorOps.Clear(_outputGradients);
        }

        public void ResetGradients()
        {
            TensorOps.Clear(_queryGradients);
            TensorOps.Clear(_keyGradients);
            TensorOps.Clear(_valueGradients);
            TensorOps.Clear(_outputGradients);
            TensorOps.Clear(_accumulatedQueryGradients);
            TensorOps.Clear(_accumulatedKeyGradients);
            TensorOps.Clear(_accumulatedValueGradients);
            TensorOps.Clear(_accumulatedOutputGradients);

            _queryLora?.ResetGradients();
            _keyLora?.ResetGradients();
            _valueLora?.ResetGradients();
            _outputLora?.ResetGradients();
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
                OutputOptimizerState = _outputOptimizer.SaveState(),
                LoraState = SaveLoraState()
            };
        }

        private float[] FlattenMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows * cols];

            // float[,] rectangular es contiguo row-major: aplanar es un memcpy
            // puro, sin aritmética, no una copia elemento a elemento.
            System.Buffer.BlockCopy(matrix, 0, result, 0, result.Length * sizeof(float));

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

            attention._queryWeightsCacheFp16.Invalidate();
            attention._keyWeightsCacheFp16.Invalidate();
            attention._valueWeightsCacheFp16.Invalidate();
            attention._outputWeightsCacheFp16.Invalidate();

            attention._queryWeightsCacheInt8.Invalidate();
            attention._keyWeightsCacheInt8.Invalidate();
            attention._valueWeightsCacheInt8.Invalidate();
            attention._outputWeightsCacheInt8.Invalidate();

            if (state.LoraState != null)
            {
                attention.LoadLoraState(state.LoraState);
            }

            return attention;
        }

        private static float[,] UnflattenMatrix(float[] array, int rows, int cols)
        {
            var matrix = new float[rows, cols];

            System.Buffer.BlockCopy(array, 0, matrix, 0, array.Length * sizeof(float));

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

        public LoraAttentionState? LoraState { get; set; }

        public MultiHeadAttentionState()
        {
            QueryWeights = Array.Empty<float>();
            KeyWeights = Array.Empty<float>();
            ValueWeights = Array.Empty<float>();
            OutputWeights = Array.Empty<float>();
        }
    }
}