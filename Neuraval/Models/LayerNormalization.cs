using Neuraval.Core.Utils;
using Neuraval.Cuda;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class LayerNormalization
    {
        private readonly int _normalizedShape;
        private readonly float _epsilon;
        private float[] _gamma;
        private float[] _beta;
        private float[] _gammaGradients;
        private float[] _betaGradients;
        private float[] _accumulatedGammaGradients;
        private float[] _accumulatedBetaGradients;
        private AdamVectorOptimizer _gammaOptimizer = null!;
        private AdamVectorOptimizer _betaOptimizer = null!;

        private float[,]? _lastInput;
        private float[]? _lastMean;
        private float[]? _lastStd;

        private float[,,]? _lastInputBatch;
        private float[,]? _lastMeanBatch;
        private float[,]? _lastStdBatch;

        public int NormalizedShape => _normalizedShape;

        public LayerNormalization(int normalizedShape, float epsilon = 1e-5f)
        {
            _normalizedShape = normalizedShape;
            _epsilon = epsilon;

            _gamma = new float[normalizedShape];
            _beta = new float[normalizedShape];
            _gammaGradients = new float[normalizedShape];
            _betaGradients = new float[normalizedShape];
            _accumulatedGammaGradients = new float[normalizedShape];
            _accumulatedBetaGradients = new float[normalizedShape];
            _gammaOptimizer = new AdamVectorOptimizer(normalizedShape);
            _betaOptimizer = new AdamVectorOptimizer(normalizedShape);

            for (int i = 0; i < normalizedShape; i++)
            {
                _gamma[i] = 1.0f;
                _beta[i] = 0.0f;
            }
        }

        public void ZeroGradients()
        {
            Array.Clear(_accumulatedGammaGradients, 0, _accumulatedGammaGradients.Length);
            Array.Clear(_accumulatedBetaGradients, 0, _accumulatedBetaGradients.Length);
        }

        public void AverageGradients(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive");

            float scale = 1.0f / batchSize;

            _gammaGradients = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray1D(_accumulatedGammaGradients), scale).ToArray1D();
            _betaGradients = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray1D(_accumulatedBetaGradients), scale).ToArray1D();
        }

        public float GammaGradientAt(int index) => _gammaGradients[index];

        public float BetaGradientAt(int index) => _betaGradients[index];

        public float SumSquaredGradients()
        {
            float total = SumSquared(Neuraval.Tensor.Tensor.FromArray1D(_gammaGradients));
            total += SumSquared(Neuraval.Tensor.Tensor.FromArray1D(_betaGradients));
            return total;
        }

        private static float SumSquared(Neuraval.Tensor.Tensor tensor)
        {
            return TensorOps.Sum(TensorOps.Multiply(tensor, tensor));
        }

        public void ScaleGradients(float scale)
        {
            _gammaGradients = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray1D(_gammaGradients), scale).ToArray1D();
            _betaGradients = TensorOps.Scale(Neuraval.Tensor.Tensor.FromArray1D(_betaGradients), scale).ToArray1D();
        }

        public float[,] Forward(float[,] input)
        {
            int dim = input.GetLength(1);

            if (dim != _normalizedShape)
            {
                throw new ArgumentException($"Input dimension {dim} does not match normalized shape {_normalizedShape}");
            }

            _lastInput = (float[,])input.Clone();

            var device = TensorDeviceSelector.Current;
            var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(input, device);
            var gammaTensor = Neuraval.Tensor.Tensor.FromArray1D(_gamma, device);
            var betaTensor = Neuraval.Tensor.Tensor.FromArray1D(_beta, device);

            try
            {
                var outputTensor = TensorOps.LayerNormRows(inputTensor, gammaTensor, betaTensor, _epsilon, out var mean, out var std);
                _lastMean = mean;
                _lastStd = std;

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
            if (_lastInput == null || _lastMean == null || _lastStd == null)
                throw new InvalidOperationException("Forward must be called before Backward");

            return BackwardCore(gradOutput, _lastInput, _lastMean, _lastStd);
        }

        private float[,] BackwardCore(float[,] gradOutput, float[,] input, float[] mean, float[] std)
        {
            int rows = gradOutput.GetLength(0);
            int dim = gradOutput.GetLength(1);

            var normalized = ComputeNormalized(input, mean, std, rows, dim);
            AccumulateGammaBetaGradients(gradOutput, normalized);

            var gradInput = new float[rows, dim];

            Parallel.For(0, rows, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                float rowStd = std[i];
                float rowMean = mean[i];
                float gradMean = 0;
                float gradVar = 0;

                for (int j = 0; j < dim; j++)
                {
                    float gradNorm = gradOutput[i, j] * _gamma[j];
                    gradVar += gradNorm * (input[i, j] - rowMean) * (-0.5f) * MathF.Pow(rowStd, -3f);
                    gradMean += gradNorm * (-1.0f / rowStd);
                }

                for (int j = 0; j < dim; j++)
                {
                    float gradNorm = gradOutput[i, j] * _gamma[j];
                    gradInput[i, j] = (gradNorm / rowStd) +
                                      (gradVar * 2 * (input[i, j] - rowMean) / dim) +
                                      (gradMean / dim);
                }
            });

            return gradInput;
        }

        private static float[,] ComputeNormalized(float[,] input, float[] mean, float[] std, int rows, int dim)
        {
            var normalized = new float[rows, dim];

            Parallel.For(0, rows, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                float rowMean = mean[i];
                float rowStd = std[i];

                for (int j = 0; j < dim; j++)
                {
                    normalized[i, j] = (input[i, j] - rowMean) / rowStd;
                }
            });

            return normalized;
        }

        private void AccumulateGammaBetaGradients(float[,] gradOutput, float[,] normalized)
        {
            var gradOutputTensor = Neuraval.Tensor.Tensor.FromArray2D(gradOutput);
            var normalizedTensor = Neuraval.Tensor.Tensor.FromArray2D(normalized);

            var gammaGrad = TensorOps.SumRows(TensorOps.Multiply(gradOutputTensor, normalizedTensor)).ToArray1D();
            var betaGrad = TensorOps.SumRows(gradOutputTensor).ToArray1D();

            for (int j = 0; j < _normalizedShape; j++)
            {
                _accumulatedGammaGradients[j] += gammaGrad[j];
                _accumulatedBetaGradients[j] += betaGrad[j];
            }
        }

        public float[,,] ForwardBatch(float[,,] inputBatch)
        {
            int batchSize = inputBatch.GetLength(0);
            int seqLen = inputBatch.GetLength(1);
            int dim = inputBatch.GetLength(2);

            if (dim != _normalizedShape)
            {
                throw new ArgumentException($"Input dimension {dim} does not match normalized shape {_normalizedShape}");
            }

            _lastInputBatch = (float[,,])inputBatch.Clone();

            var flatInput = Matematicas.FlattenBatch(inputBatch);

            var device = TensorDeviceSelector.Current;
            var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(flatInput, device);
            var gammaTensor = Neuraval.Tensor.Tensor.FromArray1D(_gamma, device);
            var betaTensor = Neuraval.Tensor.Tensor.FromArray1D(_beta, device);

            float[] flatMean;
            float[] flatStd;
            float[,] flatOutput;

            try
            {
                var outputTensor = TensorOps.LayerNormRows(inputTensor, gammaTensor, betaTensor, _epsilon, out flatMean, out flatStd);
                flatOutput = outputTensor.ToArray2D();
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return ForwardBatch(inputBatch);
            }

            _lastMeanBatch = new float[batchSize, seqLen];
            _lastStdBatch = new float[batchSize, seqLen];

            Parallel.For(0, batchSize * seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, flatIndex =>
            {
                int b = flatIndex / seqLen;
                int i = flatIndex % seqLen;
                _lastMeanBatch[b, i] = flatMean[flatIndex];
                _lastStdBatch[b, i] = flatStd[flatIndex];
            });

            return Matematicas.UnflattenBatch(flatOutput, batchSize, seqLen);
        }

        public float[,,] BackwardBatch(float[,,] gradOutputBatch, float learningRate)
        {
            if (_lastInputBatch == null || _lastMeanBatch == null || _lastStdBatch == null)
                throw new InvalidOperationException("ForwardBatch must be called before BackwardBatch");

            int batchSize = gradOutputBatch.GetLength(0);
            int seqLen = gradOutputBatch.GetLength(1);

            var flatGradOutput = Matematicas.FlattenBatch(gradOutputBatch);
            var flatInput = Matematicas.FlattenBatch(_lastInputBatch);
            var flatMean = FlattenRowStats(_lastMeanBatch, batchSize, seqLen);
            var flatStd = FlattenRowStats(_lastStdBatch, batchSize, seqLen);

            var flatGradInput = BackwardCore(flatGradOutput, flatInput, flatMean, flatStd);

            return Matematicas.UnflattenBatch(flatGradInput, batchSize, seqLen);
        }

        private static float[] FlattenRowStats(float[,] stats, int batchSize, int seqLen)
        {
            var flat = new float[batchSize * seqLen];

            // stats[,] es contiguo row-major y su forma [batchSize, seqLen] ya
            // coincide elemento a elemento con el flat de salida: memcpy puro.
            System.Buffer.BlockCopy(stats, 0, flat, 0, flat.Length * sizeof(float));

            return flat;
        }

        public void UpdateWeights(float learningRate)
        {
            _gammaOptimizer.Update(_gamma, _gammaGradients, learningRate);
            _betaOptimizer.Update(_beta, _betaGradients, learningRate);
            Array.Clear(_gammaGradients, 0, _gammaGradients.Length);
            Array.Clear(_betaGradients, 0, _betaGradients.Length);
        }

        public void ResetGradients()
        {
            Array.Clear(_gammaGradients, 0, _gammaGradients.Length);
            Array.Clear(_betaGradients, 0, _betaGradients.Length);
            Array.Clear(_accumulatedGammaGradients, 0, _accumulatedGammaGradients.Length);
            Array.Clear(_accumulatedBetaGradients, 0, _accumulatedBetaGradients.Length);
        }

        public LayerNormalizationState SaveState()
        {
            return new LayerNormalizationState
            {
                NormalizedShape = _normalizedShape,
                Epsilon = _epsilon,
                Gamma = (float[])_gamma.Clone(),
                Beta = (float[])_beta.Clone(),
                GammaOptimizerState = _gammaOptimizer.SaveState(),
                BetaOptimizerState = _betaOptimizer.SaveState()
            };
        }

        public static LayerNormalization LoadState(LayerNormalizationState state)
        {
            var layer = new LayerNormalization(state.NormalizedShape, state.Epsilon);
            layer._gamma = (float[])state.Gamma.Clone();
            layer._beta = (float[])state.Beta.Clone();

            if (state.GammaOptimizerState != null) layer._gammaOptimizer.LoadStateInto(state.GammaOptimizerState);
            if (state.BetaOptimizerState != null) layer._betaOptimizer.LoadStateInto(state.BetaOptimizerState);

            return layer;
        }
    }

    public class LayerNormalizationState
    {
        public int NormalizedShape { get; set; }
        public float Epsilon { get; set; }
        public float[] Gamma { get; set; }
        public float[] Beta { get; set; }
        public AdamVectorOptimizerState? GammaOptimizerState { get; set; }
        public AdamVectorOptimizerState? BetaOptimizerState { get; set; }

        public LayerNormalizationState()
        {
            Gamma = Array.Empty<float>();
            Beta = Array.Empty<float>();
        }
    }
}
