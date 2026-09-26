using System;
using System.Threading.Tasks;
using Neuraval.Core.Utils;
using Neuraval.Cuda;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class SwiGLUFeedForward
    {
        private readonly int _embeddingDim;
        private readonly int _hiddenDim;
        private readonly Random _random;

        private float[,] _weightsGate;
        private float[,] _weightsUp;
        private float[,] _weightsDown;

        private float[,] _gradientsGate;
        private float[,] _gradientsUp;
        private float[,] _gradientsDown;

        private float[,] _accumulatedGradientsGate;
        private float[,] _accumulatedGradientsUp;
        private float[,] _accumulatedGradientsDown;

        private AdamMatrixOptimizer _weightsGateOptimizer = null!;
        private AdamMatrixOptimizer _weightsUpOptimizer = null!;
        private AdamMatrixOptimizer _weightsDownOptimizer = null!;

        private CudaWeightCache _weightsGateCache = null!;
        private CudaWeightCache _weightsUpCache = null!;
        private CudaWeightCache _weightsDownCache = null!;

        private float[,]? _lastInput;
        private float[,]? _lastGatePre;
        private float[,]? _lastUp;
        private float[,]? _lastHidden;

        private float[,,]? _lastInputBatch;
        private float[,,]? _lastGatePreBatch;
        private float[,,]? _lastUpBatch;
        private float[,,]? _lastHiddenBatch;

        public int EmbeddingDim => _embeddingDim;
        public int HiddenDim => _hiddenDim;

        public SwiGLUFeedForward(int embeddingDim, int hiddenDim, int seed = 42)
        {
            _embeddingDim = embeddingDim;
            _hiddenDim = hiddenDim;
            _random = new Random(seed);

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            float limitGateUp = MathF.Sqrt(6.0f / (_embeddingDim + _hiddenDim));
            float limitDown = MathF.Sqrt(6.0f / (_hiddenDim + _embeddingDim));

            _weightsGate = InitializeMatrix(_embeddingDim, _hiddenDim, limitGateUp);
            _weightsUp = InitializeMatrix(_embeddingDim, _hiddenDim, limitGateUp);
            _weightsDown = InitializeMatrix(_hiddenDim, _embeddingDim, limitDown);

            _gradientsGate = new float[_embeddingDim, _hiddenDim];
            _gradientsUp = new float[_embeddingDim, _hiddenDim];
            _gradientsDown = new float[_hiddenDim, _embeddingDim];

            _accumulatedGradientsGate = new float[_embeddingDim, _hiddenDim];
            _accumulatedGradientsUp = new float[_embeddingDim, _hiddenDim];
            _accumulatedGradientsDown = new float[_hiddenDim, _embeddingDim];

            _weightsGateOptimizer = new AdamMatrixOptimizer(_embeddingDim, _hiddenDim);
            _weightsUpOptimizer = new AdamMatrixOptimizer(_embeddingDim, _hiddenDim);
            _weightsDownOptimizer = new AdamMatrixOptimizer(_hiddenDim, _embeddingDim);

            _weightsGateCache = new CudaWeightCache(_embeddingDim, _hiddenDim);
            _weightsUpCache = new CudaWeightCache(_embeddingDim, _hiddenDim);
            _weightsDownCache = new CudaWeightCache(_hiddenDim, _embeddingDim);
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
            Matematicas.ParallelClearMatrix(_accumulatedGradientsGate);
            Matematicas.ParallelClearMatrix(_accumulatedGradientsUp);
            Matematicas.ParallelClearMatrix(_accumulatedGradientsDown);
        }

        public void AverageGradients(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive");

            float scale = 1.0f / batchSize;

            _gradientsGate = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_accumulatedGradientsGate), scale).ToArray2D();
            _gradientsUp = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_accumulatedGradientsUp), scale).ToArray2D();
            _gradientsDown = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_accumulatedGradientsDown), scale).ToArray2D();
        }

        public float GateGradientAt(int i, int j) => _gradientsGate[i, j];

        public float UpGradientAt(int i, int j) => _gradientsUp[i, j];

        public float DownGradientAt(int i, int j) => _gradientsDown[i, j];

        public float SumSquaredGradients()
        {
            float total = 0f;

            total += SumSquared(Neuraval.Tensor.Tensor.FromArray2D(_gradientsGate));
            total += SumSquared(Neuraval.Tensor.Tensor.FromArray2D(_gradientsUp));
            total += SumSquared(Neuraval.Tensor.Tensor.FromArray2D(_gradientsDown));

            return total;
        }

        private static float SumSquared(Neuraval.Tensor.Tensor tensor)
        {
            return TensorOps.Sum(TensorOps.Multiply(tensor, tensor));
        }

        public void ScaleGradients(float scale)
        {
            _gradientsGate = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_gradientsGate), scale).ToArray2D();
            _gradientsUp = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_gradientsUp), scale).ToArray2D();
            _gradientsDown = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_gradientsDown), scale).ToArray2D();
        }

        private static float SiLU(float x)
        {
            float sigmoid = 1.0f / (1.0f + MathF.Exp(-x));
            return x * sigmoid;
        }

        private static float SiLUDerivative(float x)
        {
            float sigmoid = 1.0f / (1.0f + MathF.Exp(-x));
            return sigmoid * (1.0f + x * (1.0f - sigmoid));
        }

        public float[,] Forward(float[,] input)
        {
            int seqLen = input.GetLength(0);
            int embDim = input.GetLength(1);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input dimension {embDim} does not match expected {_embeddingDim}");
            }

            _lastInput = (float[,])input.Clone();

            var device = TensorDeviceSelector.Current;
            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() };

            try
            {
                var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(input, device);

                var gatePreTensor = TensorOps.MatMulCachedB(inputTensor, _weightsGate, _weightsGateCache);
                var upTensor = TensorOps.MatMulCachedB(inputTensor, _weightsUp, _weightsUpCache);
                var hiddenTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _hiddenDim }, device);

                Parallel.For(0, seqLen, parallelOptions, i =>
                {
                    int rowOffset = i * _hiddenDim;

                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        hiddenTensor.Buffer[rowOffset + j] = SiLU(gatePreTensor.Buffer[rowOffset + j]) * upTensor.Buffer[rowOffset + j];
                    }
                });

                _lastGatePre = gatePreTensor.ToArray2D();
                _lastUp = upTensor.ToArray2D();
                _lastHidden = hiddenTensor.ToArray2D();

                var outputTensor = TensorOps.MatMulCachedB(hiddenTensor, _weightsDown, _weightsDownCache);

                return outputTensor.ToArray2D();
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return Forward(input);
            }
        }

        public float[,] Backward(float[,] gradOutput, float learningRate)
        {
            return BackwardCore(gradOutput, _lastInput!, _lastGatePre!, _lastUp!, _lastHidden!);
        }

        private float[,] BackwardCore(float[,] gradOutput, float[,] input, float[,] gatePre, float[,] up, float[,] hidden)
        {
            int seqLen = input.GetLength(0);
            var device = TensorDeviceSelector.Current;
            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() };

            try
            {
                var gradOutputTensor = Neuraval.Tensor.Tensor.FromArray2D(gradOutput, device);
                var hiddenTensor = Neuraval.Tensor.Tensor.FromArray2D(hidden, device);
                var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(input, device);
                var gatePreTensor = Neuraval.Tensor.Tensor.FromArray2D(gatePre, device);
                var upTensor = Neuraval.Tensor.Tensor.FromArray2D(up, device);

                var gradDown = TensorOps.MatMulTransposeA(hiddenTensor, gradOutputTensor);
                var gradHiddenTensor = TensorOps.MatMulTransposeBCachedB(gradOutputTensor, _weightsDown, _weightsDownCache);

                var gradGatePreTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _hiddenDim }, device);
                var gradUpTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _hiddenDim }, device);

                Parallel.For(0, seqLen, parallelOptions, i =>
                {
                    int rowOffset = i * _hiddenDim;

                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        float gradHidden = gradHiddenTensor.Buffer[rowOffset + j];
                        float gate = gatePreTensor.Buffer[rowOffset + j];

                        gradUpTensor.Buffer[rowOffset + j] = gradHidden * SiLU(gate);
                        gradGatePreTensor.Buffer[rowOffset + j] = gradHidden * upTensor.Buffer[rowOffset + j] * SiLUDerivative(gate);
                    }
                });

                var gradGate = TensorOps.MatMulTransposeA(inputTensor, gradGatePreTensor);
                var gradUp = TensorOps.MatMulTransposeA(inputTensor, gradUpTensor);

                var gradInputFromGate = TensorOps.MatMulTransposeBCachedB(gradGatePreTensor, _weightsGate, _weightsGateCache);
                var gradInputFromUp = TensorOps.MatMulTransposeBCachedB(gradUpTensor, _weightsUp, _weightsUpCache);

                Matematicas.ParallelMatrixAddInPlace(_accumulatedGradientsDown, gradDown.ToArray2D());
                Matematicas.ParallelMatrixAddInPlace(_accumulatedGradientsGate, gradGate.ToArray2D());
                Matematicas.ParallelMatrixAddInPlace(_accumulatedGradientsUp, gradUp.ToArray2D());

                TensorOps.AddInPlace(gradInputFromGate, gradInputFromUp);

                return gradInputFromGate.ToArray2D();
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return BackwardCore(gradOutput, input, gatePre, up, hidden);
            }
        }

        public float[,,] ForwardBatch(float[,,] inputBatch)
        {
            int batchSize = inputBatch.GetLength(0);
            int seqLen = inputBatch.GetLength(1);
            int embDim = inputBatch.GetLength(2);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input dimension {embDim} does not match expected {_embeddingDim}");
            }

            _lastInputBatch = (float[,,])inputBatch.Clone();

            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() };
            var device = TensorDeviceSelector.Current;
            var flatInput = FlattenBatch(inputBatch);

            try
            {
                var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(flatInput, device);

                var gatePreFlat = TensorOps.MatMulCachedB(inputTensor, _weightsGate, _weightsGateCache);
                var upFlat = TensorOps.MatMulCachedB(inputTensor, _weightsUp, _weightsUpCache);
                var hiddenFlat = new Neuraval.Tensor.Tensor(new[] { batchSize * seqLen, _hiddenDim }, device);

                Parallel.For(0, batchSize * seqLen, parallelOptions, flatIndex =>
                {
                    int rowOffset = flatIndex * _hiddenDim;

                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        hiddenFlat.Buffer[rowOffset + j] = SiLU(gatePreFlat.Buffer[rowOffset + j]) * upFlat.Buffer[rowOffset + j];
                    }
                });

                _lastGatePreBatch = UnflattenBatch(gatePreFlat.ToArray2D(), batchSize, seqLen, _hiddenDim);
                _lastUpBatch = UnflattenBatch(upFlat.ToArray2D(), batchSize, seqLen, _hiddenDim);
                _lastHiddenBatch = UnflattenBatch(hiddenFlat.ToArray2D(), batchSize, seqLen, _hiddenDim);

                var outputFlat = TensorOps.MatMulCachedB(hiddenFlat, _weightsDown, _weightsDownCache);

                return UnflattenBatch(outputFlat.ToArray2D(), batchSize, seqLen, _embeddingDim);
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return ForwardBatch(inputBatch);
            }
        }

        public float[,,] BackwardBatch(float[,,] gradOutputBatch, float learningRate)
        {
            if (_lastInputBatch == null || _lastGatePreBatch == null || _lastUpBatch == null || _lastHiddenBatch == null)
                throw new InvalidOperationException("ForwardBatch must be called before BackwardBatch");

            int batchSize = gradOutputBatch.GetLength(0);
            int seqLen = gradOutputBatch.GetLength(1);

            var flatGradOutput = FlattenBatch(gradOutputBatch);
            var flatInput = FlattenBatch(_lastInputBatch);
            var flatGatePre = FlattenBatch(_lastGatePreBatch);
            var flatUp = FlattenBatch(_lastUpBatch);
            var flatHidden = FlattenBatch(_lastHiddenBatch);

            var flatGradInput = BackwardCore(flatGradOutput, flatInput, flatGatePre, flatUp, flatHidden);

            return UnflattenBatch(flatGradInput, batchSize, seqLen, _embeddingDim);
        }

        private static float[,] FlattenBatch(float[,,] batch)
        {
            int batchSize = batch.GetLength(0);
            int seqLen = batch.GetLength(1);
            int dim = batch.GetLength(2);
            var flat = new float[batchSize * seqLen, dim];

            System.Buffer.BlockCopy(batch, 0, flat, 0, batchSize * seqLen * dim * sizeof(float));

            return flat;
        }

        private static float[,,] UnflattenBatch(float[,] flat, int batchSize, int seqLen, int dim)
        {
            var batch = new float[batchSize, seqLen, dim];

            System.Buffer.BlockCopy(flat, 0, batch, 0, batchSize * seqLen * dim * sizeof(float));

            return batch;
        }

        public void UpdateWeights(float learningRate)
        {
            _weightsGateOptimizer.Update(_weightsGate, _gradientsGate, learningRate);
            _weightsUpOptimizer.Update(_weightsUp, _gradientsUp, learningRate);
            _weightsDownOptimizer.Update(_weightsDown, _gradientsDown, learningRate);

            _weightsGateCache.Invalidate();
            _weightsUpCache.Invalidate();
            _weightsDownCache.Invalidate();

            Matematicas.ParallelClearMatrix(_gradientsGate);
            Matematicas.ParallelClearMatrix(_gradientsUp);
            Matematicas.ParallelClearMatrix(_gradientsDown);
        }

        public void ResetGradients()
        {
            Matematicas.ParallelClearMatrix(_gradientsGate);
            Matematicas.ParallelClearMatrix(_gradientsUp);
            Matematicas.ParallelClearMatrix(_gradientsDown);
            Matematicas.ParallelClearMatrix(_accumulatedGradientsGate);
            Matematicas.ParallelClearMatrix(_accumulatedGradientsUp);
            Matematicas.ParallelClearMatrix(_accumulatedGradientsDown);
        }

        public SwiGLUFeedForwardState SaveState()
        {
            return new SwiGLUFeedForwardState
            {
                EmbeddingDim = _embeddingDim,
                HiddenDim = _hiddenDim,
                WeightsGate = FlattenMatrix(_weightsGate),
                WeightsUp = FlattenMatrix(_weightsUp),
                WeightsDown = FlattenMatrix(_weightsDown),
                WeightsGateOptimizerState = _weightsGateOptimizer.SaveState(),
                WeightsUpOptimizerState = _weightsUpOptimizer.SaveState(),
                WeightsDownOptimizerState = _weightsDownOptimizer.SaveState()
            };
        }

        private static float[] FlattenMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows * cols];

            System.Buffer.BlockCopy(matrix, 0, result, 0, result.Length * sizeof(float));

            return result;
        }

        public static SwiGLUFeedForward LoadState(SwiGLUFeedForwardState state)
        {
            var network = new SwiGLUFeedForward(state.EmbeddingDim, state.HiddenDim);

            network._weightsGate = UnflattenMatrix(state.WeightsGate, state.EmbeddingDim, state.HiddenDim);
            network._weightsUp = UnflattenMatrix(state.WeightsUp, state.EmbeddingDim, state.HiddenDim);
            network._weightsDown = UnflattenMatrix(state.WeightsDown, state.HiddenDim, state.EmbeddingDim);

            if (state.WeightsGateOptimizerState != null) network._weightsGateOptimizer.LoadStateInto(state.WeightsGateOptimizerState);
            if (state.WeightsUpOptimizerState != null) network._weightsUpOptimizer.LoadStateInto(state.WeightsUpOptimizerState);
            if (state.WeightsDownOptimizerState != null) network._weightsDownOptimizer.LoadStateInto(state.WeightsDownOptimizerState);

            network._weightsGateCache.Invalidate();
            network._weightsUpCache.Invalidate();
            network._weightsDownCache.Invalidate();

            return network;
        }

        private static float[,] UnflattenMatrix(float[] array, int rows, int cols)
        {
            var matrix = new float[rows, cols];

            System.Buffer.BlockCopy(array, 0, matrix, 0, array.Length * sizeof(float));

            return matrix;
        }
    }

    public class SwiGLUFeedForwardState
    {
        public int EmbeddingDim { get; set; }
        public int HiddenDim { get; set; }
        public float[] WeightsGate { get; set; }
        public float[] WeightsUp { get; set; }
        public float[] WeightsDown { get; set; }
        public AdamMatrixOptimizerState? WeightsGateOptimizerState { get; set; }
        public AdamMatrixOptimizerState? WeightsUpOptimizerState { get; set; }
        public AdamMatrixOptimizerState? WeightsDownOptimizerState { get; set; }

        public SwiGLUFeedForwardState()
        {
            WeightsGate = Array.Empty<float>();
            WeightsUp = Array.Empty<float>();
            WeightsDown = Array.Empty<float>();
        }
    }
}
