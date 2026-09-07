using Neuraval.Core.Utils;

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

            for (int i = 0; i < _normalizedShape; i++)
            {
                _gammaGradients[i] = _accumulatedGammaGradients[i] * scale;
                _betaGradients[i] = _accumulatedBetaGradients[i] * scale;
            }
        }

        public void ClipGradients(float maxNorm)
        {
            float totalNorm = 0;

            for (int i = 0; i < _normalizedShape; i++)
            {
                totalNorm += _gammaGradients[i] * _gammaGradients[i];
                totalNorm += _betaGradients[i] * _betaGradients[i];
            }

            totalNorm = MathF.Sqrt(totalNorm);

            if (totalNorm > maxNorm)
            {
                float scale = maxNorm / (totalNorm + 1e-10f);

                for (int i = 0; i < _normalizedShape; i++)
                {
                    _gammaGradients[i] *= scale;
                    _betaGradients[i] *= scale;
                }
            }
        }

        public float[,] Forward(float[,] input)
        {
            int dim = input.GetLength(1);

            if (dim != _normalizedShape)
            {
                throw new ArgumentException($"Input dimension {dim} does not match normalized shape {_normalizedShape}");
            }

            _lastInput = (float[,])input.Clone();

            var output = Matematicas.LayerNormRowsAuto(input, _gamma, _beta, _epsilon, out var mean, out var std);
            _lastMean = mean;
            _lastStd = std;

            return output;
        }

        public float[,] Backward(float[,] gradOutput, float learningRate)
        {
            if (_lastInput == null || _lastMean == null || _lastStd == null)
                throw new InvalidOperationException("Forward must be called before Backward");

            int seqLen = gradOutput.GetLength(0);
            int dim = gradOutput.GetLength(1);

            var gradInput = new float[seqLen, dim];

            for (int i = 0; i < seqLen; i++)
            {
                float mean = _lastMean[i];
                float std = _lastStd[i];

                var normalized = new float[dim];
                for (int j = 0; j < dim; j++)
                {
                    normalized[j] = (_lastInput[i, j] - mean) / std;
                }

                for (int j = 0; j < dim; j++)
                {
                    _accumulatedGammaGradients[j] += gradOutput[i, j] * normalized[j];
                    _accumulatedBetaGradients[j] += gradOutput[i, j];
                }

                float gradMean = 0;
                float gradVar = 0;

                for (int j = 0; j < dim; j++)
                {
                    float gradNorm = gradOutput[i, j] * _gamma[j];
                    gradVar += gradNorm * (_lastInput[i, j] - mean) * (-0.5f) * MathF.Pow(std, -3f);
                    gradMean += gradNorm * (-1.0f / std);
                }

                for (int j = 0; j < dim; j++)
                {
                    float gradNorm = gradOutput[i, j] * _gamma[j];
                    gradInput[i, j] = (gradNorm / std) +
                                      (gradVar * 2 * (_lastInput[i, j] - mean) / dim) +
                                      (gradMean / dim);
                }
            }

            return gradInput;
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
            var flatOutput = Matematicas.LayerNormRowsAuto(flatInput, _gamma, _beta, _epsilon, out var flatMean, out var flatStd);

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
            int dim = gradOutputBatch.GetLength(2);

            var gradInput = new float[batchSize, seqLen, dim];

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    float mean = _lastMeanBatch[b, i];
                    float std = _lastStdBatch[b, i];

                    var normalized = new float[dim];
                    for (int j = 0; j < dim; j++)
                    {
                        normalized[j] = (_lastInputBatch[b, i, j] - mean) / std;
                    }

                    for (int j = 0; j < dim; j++)
                    {
                        _accumulatedGammaGradients[j] += gradOutputBatch[b, i, j] * normalized[j];
                        _accumulatedBetaGradients[j] += gradOutputBatch[b, i, j];
                    }

                    float gradMean = 0;
                    float gradVar = 0;

                    for (int j = 0; j < dim; j++)
                    {
                        float gradNorm = gradOutputBatch[b, i, j] * _gamma[j];
                        gradVar += gradNorm * (_lastInputBatch[b, i, j] - mean) * (-0.5f) * MathF.Pow(std, -3f);
                        gradMean += gradNorm * (-1.0f / std);
                    }

                    for (int j = 0; j < dim; j++)
                    {
                        float gradNorm = gradOutputBatch[b, i, j] * _gamma[j];
                        gradInput[b, i, j] = (gradNorm / std) +
                                          (gradVar * 2 * (_lastInputBatch[b, i, j] - mean) / dim) +
                                          (gradMean / dim);
                    }
                }
            }

            return gradInput;
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
