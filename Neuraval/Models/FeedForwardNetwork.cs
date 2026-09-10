using System;
using System.Threading.Tasks;
using Neuraval.Core.Utils;
using Neuraval.Cuda;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class FeedForwardNetwork
    {
        private readonly int _embeddingDim;
        private readonly int _hiddenDim;
        private readonly Random _random;

        private float[,] _weights1;
        private float[] _bias1;
        private float[,] _weights2;
        private float[] _bias2;

        private float[,] _gradients1;
        private float[] _biasGradients1;
        private float[,] _gradients2;
        private float[] _biasGradients2;

        private float[,] _accumulatedGradients1;
        private float[] _accumulatedBiasGradients1;
        private float[,] _accumulatedGradients2;
        private float[] _accumulatedBiasGradients2;

        private AdamMatrixOptimizer _weights1Optimizer = null!;
        private AdamVectorOptimizer _bias1Optimizer = null!;
        private AdamMatrixOptimizer _weights2Optimizer = null!;
        private AdamVectorOptimizer _bias2Optimizer = null!;

        private CudaWeightCache _weights1Cache = null!;
        private CudaWeightCache _weights2Cache = null!;

        private float[,] _lastInput;
        private float[,] _lastHidden;

        private float[,,]? _lastInputBatch;
        private float[,,]? _lastHiddenBatch;

        public int EmbeddingDim => _embeddingDim;
        public int HiddenDim => _hiddenDim;

        public FeedForwardNetwork(int embeddingDim, int hiddenDim, int seed = 42)
        {
            _embeddingDim = embeddingDim;
            _hiddenDim = hiddenDim;
            _random = new Random(seed);

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            float limit1 = MathF.Sqrt(6.0f / (_embeddingDim + _hiddenDim));
            float limit2 = MathF.Sqrt(6.0f / (_hiddenDim + _embeddingDim));

            _weights1 = InitializeMatrix(_embeddingDim, _hiddenDim, limit1);
            _bias1 = new float[_hiddenDim];

            _weights2 = InitializeMatrix(_hiddenDim, _embeddingDim, limit2);
            _bias2 = new float[_embeddingDim];

            _gradients1 = new float[_embeddingDim, _hiddenDim];
            _biasGradients1 = new float[_hiddenDim];

            _gradients2 = new float[_hiddenDim, _embeddingDim];
            _biasGradients2 = new float[_embeddingDim];

            _accumulatedGradients1 = new float[_embeddingDim, _hiddenDim];
            _accumulatedBiasGradients1 = new float[_hiddenDim];

            _accumulatedGradients2 = new float[_hiddenDim, _embeddingDim];
            _accumulatedBiasGradients2 = new float[_embeddingDim];

            _weights1Optimizer = new AdamMatrixOptimizer(_embeddingDim, _hiddenDim);
            _bias1Optimizer = new AdamVectorOptimizer(_hiddenDim);
            _weights2Optimizer = new AdamMatrixOptimizer(_hiddenDim, _embeddingDim);
            _bias2Optimizer = new AdamVectorOptimizer(_embeddingDim);

            _weights1Cache = new CudaWeightCache(_embeddingDim, _hiddenDim);
            _weights2Cache = new CudaWeightCache(_hiddenDim, _embeddingDim);
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
            Matematicas.ParallelClearMatrix(_accumulatedGradients1);
            Matematicas.ParallelClearMatrix(_accumulatedGradients2);
            Array.Clear(_accumulatedBiasGradients1, 0, _accumulatedBiasGradients1.Length);
            Array.Clear(_accumulatedBiasGradients2, 0, _accumulatedBiasGradients2.Length);
        }

        public void AverageGradients(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive");

            float scale = 1.0f / batchSize;

            _gradients1 = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_accumulatedGradients1), scale).ToArray2D();
            _gradients2 = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_accumulatedGradients2), scale).ToArray2D();
            _biasGradients1 = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray1D(_accumulatedBiasGradients1), scale).ToArray1D();
            _biasGradients2 = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray1D(_accumulatedBiasGradients2), scale).ToArray1D();
        }

        public float Gradient1At(int i, int j) => _gradients1[i, j];

        public float BiasGradient1At(int i) => _biasGradients1[i];

        public float Gradient2At(int i, int j) => _gradients2[i, j];

        public float BiasGradient2At(int i) => _biasGradients2[i];

        public float SumSquaredGradients()
        {
            float total = 0f;

            total += SumSquared(Neuraval.Tensor.Tensor.FromArray2D(_gradients1));
            total += SumSquared(Neuraval.Tensor.Tensor.FromArray1D(_biasGradients1));
            total += SumSquared(Neuraval.Tensor.Tensor.FromArray2D(_gradients2));
            total += SumSquared(Neuraval.Tensor.Tensor.FromArray1D(_biasGradients2));

            return total;
        }

        private static float SumSquared(Neuraval.Tensor.Tensor tensor)
        {
            return TensorOps.Sum(TensorOps.Multiply(tensor, tensor));
        }

        public void ScaleGradients(float scale)
        {
            _gradients1 = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_gradients1), scale).ToArray2D();
            _gradients2 = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray2D(_gradients2), scale).ToArray2D();
            _biasGradients1 = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray1D(_biasGradients1), scale).ToArray1D();
            _biasGradients2 = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray1D(_biasGradients2), scale).ToArray1D();
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

                var preHidden = TensorOps.MatMulCachedB(inputTensor, _weights1, _weights1Cache);
                var hiddenTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _hiddenDim }, device);

                Parallel.For(0, seqLen, parallelOptions, i =>
                {
                    int rowOffset = i * _hiddenDim;

                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        hiddenTensor.Buffer[rowOffset + j] = ReLU(preHidden.Buffer[rowOffset + j] + _bias1[j]);
                    }
                });

                _lastHidden = hiddenTensor.ToArray2D();

                var preOutput = TensorOps.MatMulCachedB(hiddenTensor, _weights2, _weights2Cache);
                var outputTensor = new Neuraval.Tensor.Tensor(new[] { seqLen, _embeddingDim }, device);

                Parallel.For(0, seqLen, parallelOptions, i =>
                {
                    int rowOffset = i * _embeddingDim;

                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        outputTensor.Buffer[rowOffset + j] = preOutput.Buffer[rowOffset + j] + _bias2[j];
                    }
                });

                return outputTensor.ToArray2D();
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return Forward(input);
            }
        }

        private float ReLU(float x)
        {
            return MathF.Max(0, x);
        }

        public float[,] Backward(float[,] gradOutput, float learningRate)
        {
            return BackwardCore(gradOutput, _lastInput, _lastHidden);
        }

        private float[,] BackwardCore(float[,] gradOutput, float[,] input, float[,] hidden)
        {
            var device = TensorDeviceSelector.Current;

            try
            {
                var gradOutputTensor = Neuraval.Tensor.Tensor.FromArray2D(gradOutput, device);
                var hiddenTensor = Neuraval.Tensor.Tensor.FromArray2D(hidden, device);
                var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(input, device);

                var gradHiddenPre = TensorOps.MatMulTransposeBCachedB(gradOutputTensor, _weights2, _weights2Cache);
                var gradHiddenTensor = TensorOps.ReLUBackward(gradHiddenPre, hiddenTensor);

                var grad2 = TensorOps.MatMulTransposeA(hiddenTensor, gradOutputTensor);
                var biasGrad2 = TensorOps.SumRows(gradOutputTensor);

                var gradInputTensor = TensorOps.MatMulTransposeBCachedB(gradHiddenTensor, _weights1, _weights1Cache);

                var grad1 = TensorOps.MatMulTransposeA(inputTensor, gradHiddenTensor);
                var biasGrad1 = TensorOps.SumRows(gradHiddenTensor);

                Matematicas.ParallelMatrixAddInPlace(_accumulatedGradients2, grad2.ToArray2D());
                AccumulateVector(_accumulatedBiasGradients2, biasGrad2.ToArray1D());

                Matematicas.ParallelMatrixAddInPlace(_accumulatedGradients1, grad1.ToArray2D());
                AccumulateVector(_accumulatedBiasGradients1, biasGrad1.ToArray1D());

                return gradInputTensor.ToArray2D();
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return BackwardCore(gradOutput, input, hidden);
            }
        }

        private static void AccumulateVector(float[] target, float[] source)
        {
            for (int i = 0; i < target.Length; i++)
            {
                target[i] += source[i];
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

                var preHidden = TensorOps.MatMulCachedB(inputTensor, _weights1, _weights1Cache);
                var hiddenFlat = new Neuraval.Tensor.Tensor(new[] { batchSize * seqLen, _hiddenDim }, device);

                Parallel.For(0, batchSize * seqLen, parallelOptions, flatIndex =>
                {
                    int rowOffset = flatIndex * _hiddenDim;

                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        hiddenFlat.Buffer[rowOffset + j] = ReLU(preHidden.Buffer[rowOffset + j] + _bias1[j]);
                    }
                });

                var hidden = new float[batchSize, seqLen, _hiddenDim];

                Parallel.For(0, batchSize * seqLen, parallelOptions, flatIndex =>
                {
                    int b = flatIndex / seqLen;
                    int i = flatIndex % seqLen;
                    int rowOffset = flatIndex * _hiddenDim;

                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        hidden[b, i, j] = hiddenFlat.Buffer[rowOffset + j];
                    }
                });

                _lastHiddenBatch = hidden;

                var preOutput = TensorOps.MatMulCachedB(hiddenFlat, _weights2, _weights2Cache);
                var output = new float[batchSize, seqLen, _embeddingDim];

                Parallel.For(0, batchSize * seqLen, parallelOptions, flatIndex =>
                {
                    int b = flatIndex / seqLen;
                    int i = flatIndex % seqLen;
                    int rowOffset = flatIndex * _embeddingDim;

                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        output[b, i, j] = preOutput.Buffer[rowOffset + j] + _bias2[j];
                    }
                });

                return output;
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return ForwardBatch(inputBatch);
            }
        }

        private static float[,] FlattenBatch(float[,,] batch)
        {
            int batchSize = batch.GetLength(0);
            int seqLen = batch.GetLength(1);
            int dim = batch.GetLength(2);
            var flat = new float[batchSize * seqLen, dim];

            // batch[b, i, j] y flat[b*seqLen + i, j] comparten el mismo layout
            // row-major contiguo: aplanar es un único memcpy (mismo caso ya
            // corregido en TransformerModel.FlattenBatch; esta era una copia
            // separada del mismo método que había quedado sin migrar).
            System.Buffer.BlockCopy(batch, 0, flat, 0, batchSize * seqLen * dim * sizeof(float));

            return flat;
        }

        public float[,,] BackwardBatch(float[,,] gradOutputBatch, float learningRate)
        {
            if (_lastInputBatch == null || _lastHiddenBatch == null)
                throw new InvalidOperationException("ForwardBatch must be called before BackwardBatch");

            int batchSize = gradOutputBatch.GetLength(0);
            int seqLen = gradOutputBatch.GetLength(1);

            var flatGradOutput = FlattenBatch(gradOutputBatch);
            var flatInput = FlattenBatch(_lastInputBatch);
            var flatHidden = FlattenBatch(_lastHiddenBatch);

            var flatGradInput = BackwardCore(flatGradOutput, flatInput, flatHidden);

            var gradInput = new float[batchSize, seqLen, _embeddingDim];

            // flatGradInput[b*seqLen + i, j] y gradInput[b, i, j] comparten el
            // mismo layout row-major contiguo: un único memcpy, no una copia
            // elemento a elemento (mismo caso que FlattenBatch arriba).
            System.Buffer.BlockCopy(flatGradInput, 0, gradInput, 0, batchSize * seqLen * _embeddingDim * sizeof(float));

            return gradInput;
        }

        public void UpdateWeights(float learningRate)
        {
            _weights1Optimizer.Update(_weights1, _gradients1, learningRate);
            _weights2Optimizer.Update(_weights2, _gradients2, learningRate);
            _bias1Optimizer.Update(_bias1, _biasGradients1, learningRate);
            _bias2Optimizer.Update(_bias2, _biasGradients2, learningRate);

            _weights1Cache.Invalidate();
            _weights2Cache.Invalidate();

            Array.Clear(_biasGradients1, 0, _biasGradients1.Length);
            Array.Clear(_biasGradients2, 0, _biasGradients2.Length);
            Matematicas.ParallelClearMatrix(_gradients1);
            Matematicas.ParallelClearMatrix(_gradients2);
        }

        public void ResetGradients()
        {
            Matematicas.ParallelClearMatrix(_gradients1);
            Matematicas.ParallelClearMatrix(_gradients2);
            Matematicas.ParallelClearMatrix(_accumulatedGradients1);
            Matematicas.ParallelClearMatrix(_accumulatedGradients2);
            Array.Clear(_biasGradients1, 0, _biasGradients1.Length);
            Array.Clear(_biasGradients2, 0, _biasGradients2.Length);
            Array.Clear(_accumulatedBiasGradients1, 0, _accumulatedBiasGradients1.Length);
            Array.Clear(_accumulatedBiasGradients2, 0, _accumulatedBiasGradients2.Length);
        }

        public FeedForwardNetworkState SaveState()
        {
            return new FeedForwardNetworkState
            {
                EmbeddingDim = _embeddingDim,
                HiddenDim = _hiddenDim,
                Weights1 = FlattenMatrix(_weights1),
                Bias1 = (float[])_bias1.Clone(),
                Weights2 = FlattenMatrix(_weights2),
                Bias2 = (float[])_bias2.Clone(),
                Weights1OptimizerState = _weights1Optimizer.SaveState(),
                Bias1OptimizerState = _bias1Optimizer.SaveState(),
                Weights2OptimizerState = _weights2Optimizer.SaveState(),
                Bias2OptimizerState = _bias2Optimizer.SaveState()
            };
        }

        private float[] FlattenMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows * cols];

            System.Buffer.BlockCopy(matrix, 0, result, 0, result.Length * sizeof(float));

            return result;
        }

        public static FeedForwardNetwork LoadState(FeedForwardNetworkState state)
        {
            var network = new FeedForwardNetwork(state.EmbeddingDim, state.HiddenDim);

            network._weights1 = UnflattenMatrix(state.Weights1, state.EmbeddingDim, state.HiddenDim);
            network._bias1 = (float[])state.Bias1.Clone();
            network._weights2 = UnflattenMatrix(state.Weights2, state.HiddenDim, state.EmbeddingDim);
            network._bias2 = (float[])state.Bias2.Clone();

            if (state.Weights1OptimizerState != null) network._weights1Optimizer.LoadStateInto(state.Weights1OptimizerState);
            if (state.Bias1OptimizerState != null) network._bias1Optimizer.LoadStateInto(state.Bias1OptimizerState);
            if (state.Weights2OptimizerState != null) network._weights2Optimizer.LoadStateInto(state.Weights2OptimizerState);
            if (state.Bias2OptimizerState != null) network._bias2Optimizer.LoadStateInto(state.Bias2OptimizerState);

            network._weights1Cache.Invalidate();
            network._weights2Cache.Invalidate();

            return network;
        }

        private static float[,] UnflattenMatrix(float[] array, int rows, int cols)
        {
            var matrix = new float[rows, cols];

            System.Buffer.BlockCopy(array, 0, matrix, 0, array.Length * sizeof(float));

            return matrix;
        }
    }

    public class FeedForwardNetworkState
    {
        public int EmbeddingDim { get; set; }
        public int HiddenDim { get; set; }
        public float[] Weights1 { get; set; }
        public float[] Bias1 { get; set; }
        public float[] Weights2 { get; set; }
        public float[] Bias2 { get; set; }
        public AdamMatrixOptimizerState? Weights1OptimizerState { get; set; }
        public AdamVectorOptimizerState? Bias1OptimizerState { get; set; }
        public AdamMatrixOptimizerState? Weights2OptimizerState { get; set; }
        public AdamVectorOptimizerState? Bias2OptimizerState { get; set; }

        public FeedForwardNetworkState()
        {
            Weights1 = Array.Empty<float>();
            Bias1 = Array.Empty<float>();
            Weights2 = Array.Empty<float>();
            Bias2 = Array.Empty<float>();
        }
    }
}