using System;
using System.Threading.Tasks;
using Neuraval.Core.Utils;

namespace Neuraval.Core.Models
{
    public class RMSNorm
    {
        private readonly int _normalizedShape;
        private readonly float _epsilon;
        private float[] _weight;
        private float[] _weightGradients;
        private float[] _accumulatedWeightGradients;
        private AdamVectorOptimizer _weightOptimizer = null!;
        private readonly bool _inferenceOnly;

        private float[,]? _lastInput;
        private float[]? _lastRms;

        private float[,,]? _lastInputBatch;
        private float[,]? _lastRmsBatch;

        public int NormalizedShape => _normalizedShape;
        public float Epsilon => _epsilon;

        public RMSNorm(int normalizedShape, float epsilon = 1e-6f, bool inferenceOnly = false)
        {
            _normalizedShape = normalizedShape;
            _epsilon = epsilon;
            _inferenceOnly = inferenceOnly;

            _weight = new float[normalizedShape];
            _weightGradients = inferenceOnly ? Array.Empty<float>() : new float[normalizedShape];
            _accumulatedWeightGradients = inferenceOnly ? Array.Empty<float>() : new float[normalizedShape];
            if (!inferenceOnly)
                _weightOptimizer = new AdamVectorOptimizer(normalizedShape);

            for (int i = 0; i < normalizedShape; i++)
            {
                _weight[i] = 1.0f;
            }
        }

        public void ZeroGradients()
        {
            Array.Clear(_accumulatedWeightGradients, 0, _accumulatedWeightGradients.Length);
        }

        public void AverageGradients(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive");

            float scale = 1.0f / batchSize;

            for (int j = 0; j < _normalizedShape; j++)
            {
                _weightGradients[j] = _accumulatedWeightGradients[j] * scale;
            }
        }

        public float WeightGradientAt(int index) => _weightGradients[index];

        public float SumSquaredGradients()
        {
            float total = 0f;
            for (int j = 0; j < _normalizedShape; j++)
            {
                total += _weightGradients[j] * _weightGradients[j];
            }
            return total;
        }

        public void ScaleGradients(float scale)
        {
            for (int j = 0; j < _normalizedShape; j++)
            {
                _weightGradients[j] *= scale;
            }
        }

        public float[,] Forward(float[,] input)
        {
            int rows = input.GetLength(0);
            int dim = input.GetLength(1);

            if (dim != _normalizedShape)
                throw new ArgumentException($"Input dimension {dim} does not match normalized shape {_normalizedShape}");

            if (!_inferenceOnly)
                _lastInput = (float[,])input.Clone();

            var output = new float[rows, dim];
            var rms = new float[rows];

            Parallel.For(0, rows, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                float sumSquares = 0f;
                for (int j = 0; j < dim; j++)
                {
                    sumSquares += input[i, j] * input[i, j];
                }

                float rowRms = MathF.Sqrt(sumSquares / dim + _epsilon);
                rms[i] = rowRms;

                for (int j = 0; j < dim; j++)
                {
                    output[i, j] = (input[i, j] / rowRms) * _weight[j];
                }
            });

            if (!_inferenceOnly)
                _lastRms = rms;

            return output;
        }

        public float[,] Backward(float[,] gradOutput)
        {
            if (_lastInput == null || _lastRms == null)
                throw new InvalidOperationException("Forward must be called before Backward");

            return BackwardCore(gradOutput, _lastInput, _lastRms);
        }

        private float[,] BackwardCore(float[,] gradOutput, float[,] input, float[] rms)
        {
            int rows = gradOutput.GetLength(0);
            int dim = gradOutput.GetLength(1);

            var gradInput = new float[rows, dim];

            Parallel.For(0, rows, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                float rowRms = rms[i];
                float rmsCubed = rowRms * rowRms * rowRms;
                float weightedGradDotInput = 0f;

                for (int j = 0; j < dim; j++)
                {
                    weightedGradDotInput += gradOutput[i, j] * _weight[j] * input[i, j];
                }

                for (int j = 0; j < dim; j++)
                {
                    float gradNorm = gradOutput[i, j] * _weight[j];
                    gradInput[i, j] = (gradNorm / rowRms) - (input[i, j] * weightedGradDotInput) / (dim * rmsCubed);
                }
            });

            AccumulateWeightGradients(gradOutput, input, rms, rows, dim);

            return gradInput;
        }

        private void AccumulateWeightGradients(float[,] gradOutput, float[,] input, float[] rms, int rows, int dim)
        {
            var partial = new float[rows, dim];

            Parallel.For(0, rows, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, i =>
            {
                float rowRms = rms[i];
                for (int j = 0; j < dim; j++)
                {
                    partial[i, j] = gradOutput[i, j] * (input[i, j] / rowRms);
                }
            });

            for (int j = 0; j < dim; j++)
            {
                float sum = 0f;
                for (int i = 0; i < rows; i++)
                {
                    sum += partial[i, j];
                }
                _accumulatedWeightGradients[j] += sum;
            }
        }

        public float[,,] ForwardBatch(float[,,] inputBatch)
        {
            int batchSize = inputBatch.GetLength(0);
            int seqLen = inputBatch.GetLength(1);
            int dim = inputBatch.GetLength(2);

            if (dim != _normalizedShape)
                throw new ArgumentException($"Input dimension {dim} does not match normalized shape {_normalizedShape}");

            if (!_inferenceOnly)
                _lastInputBatch = (float[,,])inputBatch.Clone();

            var output = new float[batchSize, seqLen, dim];
            var rms = new float[batchSize, seqLen];

            Action<int> normalizeRow = flatIndex =>
            {
                int b = flatIndex / seqLen;
                int s = flatIndex % seqLen;

                float sumSquares = 0f;
                for (int j = 0; j < dim; j++)
                {
                    sumSquares += inputBatch[b, s, j] * inputBatch[b, s, j];
                }

                float rowRms = MathF.Sqrt(sumSquares / dim + _epsilon);
                rms[b, s] = rowRms;

                for (int j = 0; j < dim; j++)
                {
                    output[b, s, j] = (inputBatch[b, s, j] / rowRms) * _weight[j];
                }
            };

            int rowCount = batchSize * seqLen;
            if (rowCount <= 2)
            {
                for (int row = 0; row < rowCount; row++)
                    normalizeRow(row);
            }
            else
            {
                Parallel.For(0, rowCount, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, normalizeRow);
            }

            if (!_inferenceOnly)
                _lastRmsBatch = rms;

            return output;
        }

        public float[,,] BackwardBatch(float[,,] gradOutputBatch)
        {
            if (_lastInputBatch == null || _lastRmsBatch == null)
                throw new InvalidOperationException("ForwardBatch must be called before BackwardBatch");

            int batchSize = gradOutputBatch.GetLength(0);
            int seqLen = gradOutputBatch.GetLength(1);
            int dim = gradOutputBatch.GetLength(2);

            var flatGradOutput = FlattenBatch(gradOutputBatch, batchSize, seqLen, dim);
            var flatInput = FlattenBatch(_lastInputBatch, batchSize, seqLen, dim);
            var flatRms = FlattenRowStats(_lastRmsBatch, batchSize, seqLen);

            var flatGradInput = BackwardCore(flatGradOutput, flatInput, flatRms);

            return UnflattenBatch(flatGradInput, batchSize, seqLen, dim);
        }

        private static float[,] FlattenBatch(float[,,] batch, int batchSize, int seqLen, int dim)
        {
            var flat = new float[batchSize * seqLen, dim];
            Buffer.BlockCopy(batch, 0, flat, 0, batchSize * seqLen * dim * sizeof(float));
            return flat;
        }

        private static float[,,] UnflattenBatch(float[,] flat, int batchSize, int seqLen, int dim)
        {
            var batch = new float[batchSize, seqLen, dim];
            Buffer.BlockCopy(flat, 0, batch, 0, batchSize * seqLen * dim * sizeof(float));
            return batch;
        }

        private static float[] FlattenRowStats(float[,] stats, int batchSize, int seqLen)
        {
            var flat = new float[batchSize * seqLen];
            Buffer.BlockCopy(stats, 0, flat, 0, flat.Length * sizeof(float));
            return flat;
        }

        public void UpdateWeights(float learningRate)
        {
            _weightOptimizer.Update(_weight, _weightGradients, learningRate);
            Array.Clear(_weightGradients, 0, _weightGradients.Length);
        }

        public void ResetGradients()
        {
            Array.Clear(_weightGradients, 0, _weightGradients.Length);
            Array.Clear(_accumulatedWeightGradients, 0, _accumulatedWeightGradients.Length);
        }

        public RMSNormState SaveState()
        {
            return new RMSNormState
            {
                NormalizedShape = _normalizedShape,
                Epsilon = _epsilon,
                Weight = (float[])_weight.Clone(),
                WeightOptimizerState = _weightOptimizer.SaveState()
            };
        }

        public static RMSNorm LoadState(RMSNormState state, bool inferenceOnly = false)
        {
            var norm = new RMSNorm(state.NormalizedShape, state.Epsilon, inferenceOnly);
            norm._weight = inferenceOnly ? state.Weight : (float[])state.Weight.Clone();

            if (!inferenceOnly && state.WeightOptimizerState != null) norm._weightOptimizer.LoadStateInto(state.WeightOptimizerState);

            return norm;
        }
    }

    public class RMSNormState
    {
        public int NormalizedShape { get; set; }
        public float Epsilon { get; set; }
        public float[] Weight { get; set; }
        public AdamVectorOptimizerState? WeightOptimizerState { get; set; }

        public RMSNormState()
        {
            Weight = Array.Empty<float>();
        }
    }
}
