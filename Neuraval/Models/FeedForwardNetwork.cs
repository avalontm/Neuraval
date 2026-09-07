using System;
using System.Threading.Tasks;
using Neuraval.Core.Utils;
using Neuraval.Cuda;

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

            Parallel.For(0, _embeddingDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                for (int j = 0; j < _hiddenDim; j++)
                {
                    _gradients1[i, j] = _accumulatedGradients1[i, j] * scale;
                }
            });

            Parallel.For(0, _hiddenDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                _biasGradients1[i] = _accumulatedBiasGradients1[i] * scale;
                for (int j = 0; j < _embeddingDim; j++)
                {
                    _gradients2[i, j] = _accumulatedGradients2[i, j] * scale;
                }
            });

            Parallel.For(0, _embeddingDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                _biasGradients2[i] = _accumulatedBiasGradients2[i] * scale;
            });
        }

        public void ClipGradients(float maxNorm)
        {
            float totalNorm = 0;
            object lockObj = new object();

            Parallel.For(0, _embeddingDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, () => 0.0f, (i, loop, partial) =>
            {
                for (int j = 0; j < _hiddenDim; j++)
                {
                    partial += _gradients1[i, j] * _gradients1[i, j];
                }
                return partial;
            }, partial =>
            {
                lock (lockObj)
                {
                    totalNorm += partial;
                }
            });

            for (int i = 0; i < _hiddenDim; i++)
            {
                totalNorm += _biasGradients1[i] * _biasGradients1[i];
            }

            Parallel.For(0, _hiddenDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, () => 0.0f, (i, loop, partial) =>
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    partial += _gradients2[i, j] * _gradients2[i, j];
                }
                return partial;
            }, partial =>
            {
                lock (lockObj)
                {
                    totalNorm += partial;
                }
            });

            for (int i = 0; i < _embeddingDim; i++)
            {
                totalNorm += _biasGradients2[i] * _biasGradients2[i];
            }

            totalNorm = MathF.Sqrt(totalNorm);

            if (totalNorm > maxNorm)
            {
                float scale = maxNorm / (totalNorm + 1e-10f);

                Parallel.For(0, _embeddingDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
                {
                    for (int j = 0; j < _hiddenDim; j++)
                    {
                        _gradients1[i, j] *= scale;
                    }
                });

                Parallel.For(0, _hiddenDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
                {
                    _biasGradients1[i] *= scale;
                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        _gradients2[i, j] *= scale;
                    }
                });

                Parallel.For(0, _embeddingDim, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
                {
                    _biasGradients2[i] *= scale;
                });
            }
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

            var preHidden = Matematicas.MatrixMultiplyAutoCached(input, _weights1, _weights1Cache);
            var hidden = new float[seqLen, _hiddenDim];

            Parallel.For(0, seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                for (int j = 0; j < _hiddenDim; j++)
                {
                    hidden[i, j] = ReLU(preHidden[i, j] + _bias1[j]);
                }
            });

            _lastHidden = hidden;

            var preOutput = Matematicas.MatrixMultiplyAutoCached(hidden, _weights2, _weights2Cache);
            var output = new float[seqLen, _embeddingDim];

            Parallel.For(0, seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    output[i, j] = preOutput[i, j] + _bias2[j];
                }
            });

            return output;
        }

        private float ReLU(float x)
        {
            return MathF.Max(0, x);
        }

        private float ReLUDerivative(float x)
        {
            return x > 0 ? 1.0f : 0.0f;
        }

        public float[,] Backward(float[,] gradOutput, float learningRate)
        {
            int seqLen = gradOutput.GetLength(0);
            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() };

            var gradHiddenPre = Matematicas.MatrixMultiplyTransposeBAutoCached(gradOutput, _weights2, _weights2Cache);
            var gradHidden = new float[seqLen, _hiddenDim];

            Parallel.For(0, seqLen, parallelOptions, i =>
            {
                for (int j = 0; j < _hiddenDim; j++)
                {
                    gradHidden[i, j] = gradHiddenPre[i, j] * ReLUDerivative(_lastHidden[i, j]);
                }
            });

            var grad2 = Matematicas.MatrixMultiplyTransposeAAuto(_lastHidden, gradOutput);
            Matematicas.ParallelMatrixAddInPlace(_accumulatedGradients2, grad2);

            Parallel.For(0, _embeddingDim, parallelOptions, i =>
            {
                float sum = _accumulatedBiasGradients2[i];
                for (int j = 0; j < seqLen; j++)
                {
                    sum += gradOutput[j, i];
                }
                _accumulatedBiasGradients2[i] = sum;
            });

            var gradInput = Matematicas.MatrixMultiplyTransposeBAutoCached(gradHidden, _weights1, _weights1Cache);

            var grad1 = Matematicas.MatrixMultiplyTransposeAAuto(_lastInput, gradHidden);
            Matematicas.ParallelMatrixAddInPlace(_accumulatedGradients1, grad1);

            Parallel.For(0, _hiddenDim, parallelOptions, i =>
            {
                float sum = _accumulatedBiasGradients1[i];
                for (int j = 0; j < seqLen; j++)
                {
                    sum += gradHidden[j, i];
                }
                _accumulatedBiasGradients1[i] = sum;
            });

            return gradInput;
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

            var flatInput = FlattenBatch(inputBatch);
            var preHiddenFlat = Matematicas.MatrixMultiplyAutoCached(flatInput, _weights1, _weights1Cache);

            var hidden = new float[batchSize, seqLen, _hiddenDim];

            Parallel.For(0, batchSize * seqLen, parallelOptions, flatIndex =>
            {
                int b = flatIndex / seqLen;
                int i = flatIndex % seqLen;

                for (int j = 0; j < _hiddenDim; j++)
                {
                    hidden[b, i, j] = ReLU(preHiddenFlat[flatIndex, j] + _bias1[j]);
                }
            });

            _lastHiddenBatch = hidden;

            var flatHidden = FlattenBatch(hidden);
            var preOutputFlat = Matematicas.MatrixMultiplyAutoCached(flatHidden, _weights2, _weights2Cache);

            var output = new float[batchSize, seqLen, _embeddingDim];

            Parallel.For(0, batchSize * seqLen, parallelOptions, flatIndex =>
            {
                int b = flatIndex / seqLen;
                int i = flatIndex % seqLen;

                for (int j = 0; j < _embeddingDim; j++)
                {
                    output[b, i, j] = preOutputFlat[flatIndex, j] + _bias2[j];
                }
            });

            return output;
        }

        private static float[,] FlattenBatch(float[,,] batch)
        {
            int batchSize = batch.GetLength(0);
            int seqLen = batch.GetLength(1);
            int dim = batch.GetLength(2);
            var flat = new float[batchSize * seqLen, dim];

            Parallel.For(0, batchSize * seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, flatIndex =>
            {
                int b = flatIndex / seqLen;
                int i = flatIndex % seqLen;

                for (int j = 0; j < dim; j++)
                {
                    flat[flatIndex, j] = batch[b, i, j];
                }
            });

            return flat;
        }

        public float[,,] BackwardBatch(float[,,] gradOutputBatch, float learningRate)
        {
            if (_lastInputBatch == null || _lastHiddenBatch == null)
                throw new InvalidOperationException("ForwardBatch must be called before BackwardBatch");

            var lastInputBatch = _lastInputBatch;
            var lastHiddenBatch = _lastHiddenBatch;

            int batchSize = gradOutputBatch.GetLength(0);
            int seqLen = gradOutputBatch.GetLength(1);
            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() };

            var flatGradOutput = FlattenBatch(gradOutputBatch);
            var gradHiddenPreFlat = Matematicas.MatrixMultiplyTransposeBAutoCached(flatGradOutput, _weights2, _weights2Cache);

            var gradHidden = new float[batchSize, seqLen, _hiddenDim];

            Parallel.For(0, batchSize * seqLen, parallelOptions, flatIndex =>
            {
                int b = flatIndex / seqLen;
                int i = flatIndex % seqLen;

                for (int j = 0; j < _hiddenDim; j++)
                {
                    gradHidden[b, i, j] = gradHiddenPreFlat[flatIndex, j] * ReLUDerivative(lastHiddenBatch[b, i, j]);
                }
            });

            var flatHidden = FlattenBatch(lastHiddenBatch);
            var grad2 = Matematicas.MatrixMultiplyTransposeAAuto(flatHidden, flatGradOutput);
            Matematicas.ParallelMatrixAddInPlace(_accumulatedGradients2, grad2);

            Parallel.For(0, _embeddingDim, parallelOptions, i =>
            {
                float sum = _accumulatedBiasGradients2[i];
                for (int b = 0; b < batchSize; b++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        sum += gradOutputBatch[b, j, i];
                    }
                }
                _accumulatedBiasGradients2[i] = sum;
            });

            var flatGradHidden = FlattenBatch(gradHidden);
            var gradInputFlat = Matematicas.MatrixMultiplyTransposeBAutoCached(flatGradHidden, _weights1, _weights1Cache);

            var gradInput = new float[batchSize, seqLen, _embeddingDim];

            Parallel.For(0, batchSize * seqLen, parallelOptions, flatIndex =>
            {
                int b = flatIndex / seqLen;
                int i = flatIndex % seqLen;

                for (int j = 0; j < _embeddingDim; j++)
                {
                    gradInput[b, i, j] = gradInputFlat[flatIndex, j];
                }
            });

            var flatInput = FlattenBatch(lastInputBatch);
            var grad1 = Matematicas.MatrixMultiplyTransposeAAuto(flatInput, flatGradHidden);
            Matematicas.ParallelMatrixAddInPlace(_accumulatedGradients1, grad1);

            Parallel.For(0, _hiddenDim, parallelOptions, i =>
            {
                float sum = _accumulatedBiasGradients1[i];
                for (int b = 0; b < batchSize; b++)
                {
                    for (int j = 0; j < seqLen; j++)
                    {
                        sum += gradHidden[b, j, i];
                    }
                }
                _accumulatedBiasGradients1[i] = sum;
            });

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