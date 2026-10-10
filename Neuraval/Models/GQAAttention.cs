using System;
using Neuraval.Cuda;
using Neuraval.Core.Models.RoPE;
using Neuraval.Core.Utils;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class GQAAttention
    {
        private readonly int _hiddenSize;
        private readonly int _numAttentionHeads;
        private readonly int _numKeyValueHeads;
        private readonly int _headDim;
        private readonly int _numGroups;
        private readonly int _kvDim;
        private readonly RotaryEmbedding? _rotary;
        private readonly bool _inferenceOnly;

        private float[,] _wq;
        private float[,] _wk;
        private float[,] _wv;
        private float[,] _wo;
        private readonly CudaWeightCache _wqkvCache;
        private readonly CudaWeightCache _wqCache;
        private readonly CudaWeightCache _wkCache;
        private readonly CudaWeightCache _wvCache;
        private readonly CudaWeightCache _woCache;
        private float[,]? _combinedQkvWeights;
        private QuantizedMatrixQ8? _wqQ8;
        private QuantizedMatrixQ8? _wkQ8;
        private QuantizedMatrixQ8? _wvQ8;
        private QuantizedMatrixQ8? _woQ8;
        private float[] _bq;
        private float[] _bk;
        private float[] _bv;

        private float[,] _wqGradients;
        private float[,] _wkGradients;
        private float[,] _wvGradients;
        private float[,] _woGradients;

        private AdamMatrixOptimizer _wqOptimizer = null!;
        private AdamMatrixOptimizer _wkOptimizer = null!;
        private AdamMatrixOptimizer _wvOptimizer = null!;
        private AdamMatrixOptimizer _woOptimizer = null!;

        private float[,,]? _lastInput;
        private float[,,,]? _lastQRotated;
        private float[,,,]? _lastKRotated;
        private float[,,,]? _lastV;
        private float[,,,]? _lastProbabilities;
        private float[,,]? _lastContextConcat;
        private int _lastPositionOffset;

        public int HiddenSize => _hiddenSize;
        public int NumAttentionHeads => _numAttentionHeads;
        public int NumKeyValueHeads => _numKeyValueHeads;
        public int HeadDim => _headDim;

        public GQAAttention(int hiddenSize, int numAttentionHeads, int numKeyValueHeads, RotaryEmbedding? rotary = null, int seed = 42, bool inferenceOnly = false)
        {
            if (hiddenSize % numAttentionHeads != 0)
                throw new ArgumentException($"hiddenSize ({hiddenSize}) debe ser divisible por numAttentionHeads ({numAttentionHeads})");

            if (numAttentionHeads % numKeyValueHeads != 0)
                throw new ArgumentException($"numAttentionHeads ({numAttentionHeads}) debe ser divisible por numKeyValueHeads ({numKeyValueHeads})");

            _hiddenSize = hiddenSize;
            _numAttentionHeads = numAttentionHeads;
            _numKeyValueHeads = numKeyValueHeads;
            _headDim = hiddenSize / numAttentionHeads;
            _numGroups = numAttentionHeads / numKeyValueHeads;
            _kvDim = numKeyValueHeads * _headDim;
            _rotary = rotary;
            _inferenceOnly = inferenceOnly;
            _bq = new float[hiddenSize];
            _bk = new float[_kvDim];
            _bv = new float[_kvDim];

            var random = new Random(seed);
            float qLimit = MathF.Sqrt(6f / (hiddenSize + hiddenSize));
            float kvLimit = MathF.Sqrt(6f / (hiddenSize + _kvDim));

            _wq = inferenceOnly ? null! : InitializeMatrix(hiddenSize, hiddenSize, qLimit, random);
            _wk = inferenceOnly ? null! : InitializeMatrix(hiddenSize, _kvDim, kvLimit, random);
            _wv = inferenceOnly ? null! : InitializeMatrix(hiddenSize, _kvDim, kvLimit, random);
            _wo = inferenceOnly ? null! : InitializeMatrix(hiddenSize, hiddenSize, qLimit, random);

            _wqkvCache = new CudaWeightCache(hiddenSize, hiddenSize + (2 * _kvDim));
            _wqCache = new CudaWeightCache(hiddenSize, hiddenSize);
            _wkCache = new CudaWeightCache(hiddenSize, _kvDim);
            _wvCache = new CudaWeightCache(hiddenSize, _kvDim);
            _woCache = new CudaWeightCache(hiddenSize, hiddenSize);

            _wqGradients = inferenceOnly ? new float[0, 0] : new float[hiddenSize, hiddenSize];
            _wkGradients = inferenceOnly ? new float[0, 0] : new float[hiddenSize, _kvDim];
            _wvGradients = inferenceOnly ? new float[0, 0] : new float[hiddenSize, _kvDim];
            _woGradients = inferenceOnly ? new float[0, 0] : new float[hiddenSize, hiddenSize];

            if (!inferenceOnly)
            {
                _wqOptimizer = new AdamMatrixOptimizer(hiddenSize, hiddenSize);
                _wkOptimizer = new AdamMatrixOptimizer(hiddenSize, _kvDim);
                _wvOptimizer = new AdamMatrixOptimizer(hiddenSize, _kvDim);
                _woOptimizer = new AdamMatrixOptimizer(hiddenSize, hiddenSize);
            }
        }

        private static float[,] InitializeMatrix(int rows, int cols, float limit, Random random)
        {
            var matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = (float)(random.NextDouble() * 2 - 1) * limit;
            return matrix;
        }

        public float[,,] Forward(float[,,] input, int positionOffset = 0)
        {
            int batchSize = input.GetLength(0);
            int seqLen = input.GetLength(1);

            if (input.GetLength(2) != _hiddenSize)
                throw new ArgumentException($"La última dimensión de entrada ({input.GetLength(2)}) no coincide con hiddenSize ({_hiddenSize})");

            if (!_inferenceOnly)
                _lastInput = (float[,,])input.Clone();
            _lastPositionOffset = positionOffset;

            var (q, k, v) = ProjectQkvHeads(input, batchSize, seqLen);

            if (_rotary != null)
            {
                q = _rotary.Apply(q, positionOffset);
                k = _rotary.Apply(k, positionOffset);
            }

            if (!_inferenceOnly)
            {
                _lastQRotated = q;
                _lastKRotated = k;
                _lastV = v;
            }

            var probabilities = _inferenceOnly ? null : new float[batchSize, _numAttentionHeads, seqLen, seqLen];
            var contextConcat = new float[batchSize, seqLen, _hiddenSize];

            for (int b = 0; b < batchSize; b++)
            {
                for (int qh = 0; qh < _numAttentionHeads; qh++)
                {
                    int kvh = qh / _numGroups;
                    float scale = 1f / MathF.Sqrt(_headDim);

                    for (int i = 0; i < seqLen; i++)
                    {
                        var scores = new float[i + 1];
                        float maxScore = float.NegativeInfinity;

                        for (int j = 0; j <= i; j++)
                        {
                            float dot = 0f;
                            for (int d = 0; d < _headDim; d++)
                                dot += q[b, i, qh, d] * k[b, j, kvh, d];

                            scores[j] = dot * scale;
                            if (scores[j] > maxScore) maxScore = scores[j];
                        }

                        float sumExp = 0f;
                        for (int j = 0; j <= i; j++)
                        {
                            scores[j] = MathF.Exp(scores[j] - maxScore);
                            sumExp += scores[j];
                        }

                        for (int j = 0; j <= i; j++)
                        {
                            float p = scores[j] / sumExp;
                            if (probabilities != null)
                                probabilities[b, qh, i, j] = p;

                            for (int d = 0; d < _headDim; d++)
                                contextConcat[b, i, qh * _headDim + d] += p * v[b, j, kvh, d];
                        }
                    }
                }
            }

            if (!_inferenceOnly)
            {
                _lastProbabilities = probabilities;
                _lastContextConcat = contextConcat;
            }

            return _inferenceOnly
                ? MatMulBatch(contextConcat, _woQ8!, batchSize, seqLen)
                : MatMulBatch(contextConcat, _wo, _woCache, batchSize, seqLen, _hiddenSize, _hiddenSize);
        }

        private (float[,,,] Queries, float[,,,] Keys, float[,,,] Values) ProjectQkvHeads(float[,,] input, int batchSize, int seqLen)
        {
            float[,,] flatOut;
            int combinedDim = _hiddenSize + (2 * _kvDim);
            if (_inferenceOnly)
            {
                var q = MatMulBatch(input, _wqQ8!, batchSize, seqLen);
                var k = MatMulBatch(input, _wkQ8!, batchSize, seqLen);
                var v = MatMulBatch(input, _wvQ8!, batchSize, seqLen);
                flatOut = new float[batchSize, seqLen, combinedDim];
                int rowBytes = combinedDim * sizeof(float);
                int projectedQueryBytes = _hiddenSize * sizeof(float);
                int projectedKvBytes = _kvDim * sizeof(float);
                for (int b = 0; b < batchSize; b++)
                    for (int s = 0; s < seqLen; s++)
                    {
                        int row = b * seqLen + s;
                        Buffer.BlockCopy(q, row * projectedQueryBytes, flatOut, row * rowBytes, projectedQueryBytes);
                        Buffer.BlockCopy(k, row * projectedKvBytes, flatOut, row * rowBytes + projectedQueryBytes, projectedKvBytes);
                        Buffer.BlockCopy(v, row * projectedKvBytes, flatOut, row * rowBytes + projectedQueryBytes + projectedKvBytes, projectedKvBytes);
                    }
            }
            else
            {
                var combinedWeights = _combinedQkvWeights ??= CombineQkvWeights();
                flatOut = MatMulBatch(input, combinedWeights, _wqkvCache, batchSize, seqLen, _hiddenSize, combinedWeights.GetLength(1));
            }
            AddQkvBias(flatOut);
            var queries = new float[batchSize, seqLen, _numAttentionHeads, _headDim];
            var keys = new float[batchSize, seqLen, _numKeyValueHeads, _headDim];
            var values = new float[batchSize, seqLen, _numKeyValueHeads, _headDim];
            int queryBytes = _hiddenSize * sizeof(float);
            int kvBytes = _kvDim * sizeof(float);
            for (int b = 0; b < batchSize; b++)
                for (int s = 0; s < seqLen; s++)
                {
                    int row = b * seqLen + s;
                    Buffer.BlockCopy(flatOut, row * combinedDim * sizeof(float), queries, row * queryBytes, queryBytes);
                    Buffer.BlockCopy(flatOut, (row * combinedDim + _hiddenSize) * sizeof(float), keys, row * kvBytes, kvBytes);
                    Buffer.BlockCopy(flatOut, (row * combinedDim + _hiddenSize + _kvDim) * sizeof(float), values, row * kvBytes, kvBytes);
                }

            return (queries, keys, values);
        }

        private float[,] CombineQkvWeights()
        {
            var combined = new float[_hiddenSize, _hiddenSize + (2 * _kvDim)];
            for (int input = 0; input < _hiddenSize; input++)
            {
                for (int output = 0; output < _hiddenSize; output++)
                    combined[input, output] = _wq[input, output];

                for (int output = 0; output < _kvDim; output++)
                {
                    combined[input, _hiddenSize + output] = _wk[input, output];
                    combined[input, _hiddenSize + _kvDim + output] = _wv[input, output];
                }
            }

            return combined;
        }

        private void AddQkvBias(float[,,] projected)
        {
            for (int batch = 0; batch < projected.GetLength(0); batch++)
                for (int token = 0; token < projected.GetLength(1); token++)
            {
                for (int i = 0; i < _hiddenSize; i++)
                    projected[batch, token, i] += _bq[i];
                for (int i = 0; i < _kvDim; i++)
                {
                    projected[batch, token, _hiddenSize + i] += _bk[i];
                    projected[batch, token, _hiddenSize + _kvDim + i] += _bv[i];
                }
            }
        }

        public float[,,] ForwardIncremental(float[,,] input, int positionOffset, GqaKeyValueCacheLayer cache)
        {
            if (cache == null)
                throw new ArgumentNullException(nameof(cache));

            int batchSize = input.GetLength(0);
            int newSeqLen = input.GetLength(1);

            if (input.GetLength(2) != _hiddenSize)
                throw new ArgumentException($"La última dimensión de entrada ({input.GetLength(2)}) no coincide con hiddenSize ({_hiddenSize})");

            if (cache.NumKeyValueHeads != _numKeyValueHeads || cache.HeadDim != _headDim)
                throw new ArgumentException("El cache no es compatible con esta capa de atención");

            var (q, k, v) = ProjectQkvHeads(input, batchSize, newSeqLen);

            if (_rotary != null)
            {
                q = _rotary.Apply(q, positionOffset);
                k = _rotary.Apply(k, positionOffset);
            }

            cache.Append(k, v);

            var contextConcat = new float[batchSize, newSeqLen, _hiddenSize];
            float scale = 1f / MathF.Sqrt(_headDim);
            var scores = new float[cache.Length];

            for (int b = 0; b < batchSize; b++)
            {
                for (int qh = 0; qh < _numAttentionHeads; qh++)
                {
                    int kvh = qh / _numGroups;

                    for (int i = 0; i < newSeqLen; i++)
                    {
                        int allowedLength = positionOffset + i + 1;
                        float maxScore = float.NegativeInfinity;

                        for (int j = 0; j < allowedLength; j++)
                        {
                            float dot = 0f;
                            for (int d = 0; d < _headDim; d++)
                                dot += q[b, i, qh, d] * cache.KeyAt(b, j, kvh, d);

                            scores[j] = dot * scale;
                            if (scores[j] > maxScore) maxScore = scores[j];
                        }

                        float sumExp = 0f;
                        for (int j = 0; j < allowedLength; j++)
                        {
                            scores[j] = MathF.Exp(scores[j] - maxScore);
                            sumExp += scores[j];
                        }

                        for (int j = 0; j < allowedLength; j++)
                        {
                            float p = scores[j] / sumExp;

                            for (int d = 0; d < _headDim; d++)
                                contextConcat[b, i, qh * _headDim + d] += p * cache.ValueAt(b, j, kvh, d);
                        }
                    }
                }
            }

            return _inferenceOnly
                ? MatMulBatch(contextConcat, _woQ8!, batchSize, newSeqLen)
                : MatMulBatch(contextConcat, _wo, _woCache, batchSize, newSeqLen, _hiddenSize, _hiddenSize);
        }

        private static float[,,] MatMulBatch(float[,,] input, QuantizedMatrixQ8 weights, int batchSize, int seqLen)
        {
            int inputDim = input.GetLength(2);
            var flatInput = new float[batchSize * seqLen, inputDim];
            Buffer.BlockCopy(input, 0, flatInput, 0, batchSize * seqLen * inputDim * sizeof(float));
            var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(flatInput, DeviceType.Cpu);
            var flatOutput = weights.Multiply(inputTensor).ToArray2D();
            var output = new float[batchSize, seqLen, weights.OutputSize];
            Buffer.BlockCopy(flatOutput, 0, output, 0, batchSize * seqLen * weights.OutputSize * sizeof(float));
            return output;
        }

        private static float[,,] MatMulBatch(float[,,] input, float[,] weight, CudaWeightCache cache, int batchSize, int seqLen, int inDim, int outDim)
        {
            var flatInput = new float[batchSize * seqLen, inDim];
            Buffer.BlockCopy(input, 0, flatInput, 0, batchSize * seqLen * inDim * sizeof(float));
            var device = TensorDeviceSelector.Current;

            try
            {
                var inputTensor = Neuraval.Tensor.Tensor.FromArray2D(flatInput, device);
                var result = TensorOps.MatMulCachedB(inputTensor, weight, cache).ToArray2D();
                var output = new float[batchSize, seqLen, outDim];
                Buffer.BlockCopy(result, 0, output, 0, batchSize * seqLen * outDim * sizeof(float));
                return output;
            }
            catch (CudaException) when (device == DeviceType.Cuda)
            {
                TensorDeviceSelector.ReportFailure();
                return MatMulBatch(input, weight, cache, batchSize, seqLen, inDim, outDim);
            }
        }

        public float[,,] Backward(float[,,] gradOutput)
        {
            if (_lastInput == null || _lastQRotated == null || _lastKRotated == null || _lastV == null || _lastProbabilities == null || _lastContextConcat == null)
                throw new InvalidOperationException("Forward must be called before Backward");

            int batchSize = _lastInput.GetLength(0);
            int seqLen = _lastInput.GetLength(1);

            var gradContextConcat = new float[batchSize, seqLen, _hiddenSize];
            AccumulateOutputProjectionGradients(gradOutput, _lastContextConcat, gradContextConcat, batchSize, seqLen);

            var gradQ = new float[batchSize, seqLen, _numAttentionHeads, _headDim];
            var gradK = new float[batchSize, seqLen, _numKeyValueHeads, _headDim];
            var gradV = new float[batchSize, seqLen, _numKeyValueHeads, _headDim];

            for (int b = 0; b < batchSize; b++)
            {
                for (int qh = 0; qh < _numAttentionHeads; qh++)
                {
                    int kvh = qh / _numGroups;
                    float scale = 1f / MathF.Sqrt(_headDim);

                    for (int i = 0; i < seqLen; i++)
                    {
                        var gradContextRow = new float[_headDim];
                        for (int d = 0; d < _headDim; d++)
                            gradContextRow[d] = gradContextConcat[b, i, qh * _headDim + d];

                        var gradP = new float[i + 1];
                        for (int j = 0; j <= i; j++)
                        {
                            float dot = 0f;
                            for (int d = 0; d < _headDim; d++)
                                dot += gradContextRow[d] * _lastV[b, j, kvh, d];
                            gradP[j] = dot;
                        }

                        float weightedSum = 0f;
                        for (int j = 0; j <= i; j++)
                            weightedSum += gradP[j] * _lastProbabilities[b, qh, i, j];

                        var gradScores = new float[i + 1];
                        for (int j = 0; j <= i; j++)
                        {
                            float p = _lastProbabilities[b, qh, i, j];
                            gradScores[j] = p * (gradP[j] - weightedSum);
                        }

                        for (int j = 0; j <= i; j++)
                        {
                            float p = _lastProbabilities[b, qh, i, j];
                            for (int d = 0; d < _headDim; d++)
                                gradV[b, j, kvh, d] += p * gradContextRow[d];

                            float gs = gradScores[j] * scale;
                            for (int d = 0; d < _headDim; d++)
                            {
                                gradQ[b, i, qh, d] += gs * _lastKRotated[b, j, kvh, d];
                                gradK[b, j, kvh, d] += gs * _lastQRotated[b, i, qh, d];
                            }
                        }
                    }
                }
            }

            if (_rotary != null)
            {
                gradQ = _rotary.ApplyBackward(gradQ, _lastPositionOffset);
                gradK = _rotary.ApplyBackward(gradK, _lastPositionOffset);
            }

            var gradInput = new float[batchSize, seqLen, _hiddenSize];

            BackpropagateProjection(gradQ, _wq, _numAttentionHeads, _lastInput, _wqGradients, gradInput, batchSize, seqLen);
            BackpropagateProjection(gradK, _wk, _numKeyValueHeads, _lastInput, _wkGradients, gradInput, batchSize, seqLen);
            BackpropagateProjection(gradV, _wv, _numKeyValueHeads, _lastInput, _wvGradients, gradInput, batchSize, seqLen);

            return gradInput;
        }

        private void AccumulateOutputProjectionGradients(float[,,] gradOutput, float[,,] contextConcat, float[,,] gradContextConcat, int batchSize, int seqLen)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int o = 0; o < _hiddenSize; o++)
                    {
                        float g = gradOutput[b, s, o];

                        for (int i = 0; i < _hiddenSize; i++)
                        {
                            _woGradients[i, o] += contextConcat[b, s, i] * g;
                            gradContextConcat[b, s, i] += g * _wo[i, o];
                        }
                    }
                }
            }
        }

        private void BackpropagateProjection(float[,,,] gradHeads, float[,] weight, int numHeads, float[,,] input, float[,] weightGradients, float[,,] gradInput, int batchSize, int seqLen)
        {
            int outDim = weight.GetLength(1);

            var gradFlat = new float[batchSize, seqLen, outDim];
            for (int b = 0; b < batchSize; b++)
                for (int s = 0; s < seqLen; s++)
                    for (int h = 0; h < numHeads; h++)
                        for (int d = 0; d < _headDim; d++)
                            gradFlat[b, s, h * _headDim + d] = gradHeads[b, s, h, d];

            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int o = 0; o < outDim; o++)
                    {
                        float g = gradFlat[b, s, o];

                        for (int i = 0; i < _hiddenSize; i++)
                        {
                            weightGradients[i, o] += input[b, s, i] * g;
                            gradInput[b, s, i] += g * weight[i, o];
                        }
                    }
                }
            }
        }

        public void ZeroGradients()
        {
            Array.Clear(_wqGradients, 0, _wqGradients.Length);
            Array.Clear(_wkGradients, 0, _wkGradients.Length);
            Array.Clear(_wvGradients, 0, _wvGradients.Length);
            Array.Clear(_woGradients, 0, _woGradients.Length);
        }

        public void AverageGradients(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive");

            float scale = 1f / batchSize;
            Scale(_wqGradients, scale);
            Scale(_wkGradients, scale);
            Scale(_wvGradients, scale);
            Scale(_woGradients, scale);
        }

        private static void Scale(float[,] matrix, float scale)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] *= scale;
        }

        public void UpdateWeights(float learningRate)
        {
            _wqOptimizer.Update(_wq, _wqGradients, learningRate);
            _wkOptimizer.Update(_wk, _wkGradients, learningRate);
            _wvOptimizer.Update(_wv, _wvGradients, learningRate);
            _woOptimizer.Update(_wo, _woGradients, learningRate);
            _combinedQkvWeights = null;
            _wqkvCache.Invalidate();
            _woCache.Invalidate();
            ZeroGradients();
        }

        public GQAAttentionState SaveState()
        {
            return new GQAAttentionState
            {
                HiddenSize = _hiddenSize,
                NumAttentionHeads = _numAttentionHeads,
                NumKeyValueHeads = _numKeyValueHeads,
                Wq = (float[,])_wq.Clone(),
                Wk = (float[,])_wk.Clone(),
                Wv = (float[,])_wv.Clone(),
                Wo = (float[,])_wo.Clone(),
                Bq = (float[])_bq.Clone(),
                Bk = (float[])_bk.Clone(),
                Bv = (float[])_bv.Clone(),
                WqOptimizerState = _wqOptimizer.SaveState(),
                WkOptimizerState = _wkOptimizer.SaveState(),
                WvOptimizerState = _wvOptimizer.SaveState(),
                WoOptimizerState = _woOptimizer.SaveState()
            };
        }

        public static GQAAttention LoadState(GQAAttentionState state, RotaryEmbedding? rotary = null, bool inferenceOnly = false)
        {
            var attention = new GQAAttention(state.HiddenSize, state.NumAttentionHeads, state.NumKeyValueHeads, rotary, inferenceOnly: inferenceOnly);
            if (inferenceOnly)
            {
                attention._wqQ8 = QuantizedMatrixQ8.FromInputOutput(state.Wq);
                attention._wkQ8 = QuantizedMatrixQ8.FromInputOutput(state.Wk);
                attention._wvQ8 = QuantizedMatrixQ8.FromInputOutput(state.Wv);
                attention._woQ8 = QuantizedMatrixQ8.FromInputOutput(state.Wo);
                attention._wq = null!;
                attention._wk = null!;
                attention._wv = null!;
                attention._wo = null!;
            }
            else
            {
                attention._wq = (float[,])state.Wq.Clone();
                attention._wk = (float[,])state.Wk.Clone();
                attention._wv = (float[,])state.Wv.Clone();
                attention._wo = (float[,])state.Wo.Clone();
            }
            if (state.Bq.Length == attention._bq.Length) attention._bq = inferenceOnly ? state.Bq : (float[])state.Bq.Clone();
            if (state.Bk.Length == attention._bk.Length) attention._bk = inferenceOnly ? state.Bk : (float[])state.Bk.Clone();
            if (state.Bv.Length == attention._bv.Length) attention._bv = inferenceOnly ? state.Bv : (float[])state.Bv.Clone();
            attention._combinedQkvWeights = null;
            attention._wqkvCache.Invalidate();
            attention._woCache.Invalidate();

            if (!inferenceOnly && state.WqOptimizerState != null) attention._wqOptimizer.LoadStateInto(state.WqOptimizerState);
            if (!inferenceOnly && state.WkOptimizerState != null) attention._wkOptimizer.LoadStateInto(state.WkOptimizerState);
            if (!inferenceOnly && state.WvOptimizerState != null) attention._wvOptimizer.LoadStateInto(state.WvOptimizerState);
            if (!inferenceOnly && state.WoOptimizerState != null) attention._woOptimizer.LoadStateInto(state.WoOptimizerState);

            return attention;
        }
    }

    public class GQAAttentionState
    {
        public int HiddenSize { get; set; }
        public int NumAttentionHeads { get; set; }
        public int NumKeyValueHeads { get; set; }
        public float[,] Wq { get; set; }
        public float[,] Wk { get; set; }
        public float[,] Wv { get; set; }
        public float[,] Wo { get; set; }
        public float[] Bq { get; set; } = Array.Empty<float>();
        public float[] Bk { get; set; } = Array.Empty<float>();
        public float[] Bv { get; set; } = Array.Empty<float>();
        public AdamMatrixOptimizerState? WqOptimizerState { get; set; }
        public AdamMatrixOptimizerState? WkOptimizerState { get; set; }
        public AdamMatrixOptimizerState? WvOptimizerState { get; set; }
        public AdamMatrixOptimizerState? WoOptimizerState { get; set; }

        public GQAAttentionState()
        {
            Wq = new float[0, 0];
            Wk = new float[0, 0];
            Wv = new float[0, 0];
            Wo = new float[0, 0];
        }
    }
}
