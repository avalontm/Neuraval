using System;
using Neuraval.Core.Models.RoPE;
using Neuraval.Core.Utils;

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

        private float[,] _wq;
        private float[,] _wk;
        private float[,] _wv;
        private float[,] _wo;

        private float[,] _wqGradients;
        private float[,] _wkGradients;
        private float[,] _wvGradients;
        private float[,] _woGradients;

        private AdamMatrixOptimizer _wqOptimizer;
        private AdamMatrixOptimizer _wkOptimizer;
        private AdamMatrixOptimizer _wvOptimizer;
        private AdamMatrixOptimizer _woOptimizer;

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

        public GQAAttention(int hiddenSize, int numAttentionHeads, int numKeyValueHeads, RotaryEmbedding? rotary = null, int seed = 42)
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

            var random = new Random(seed);
            float qLimit = MathF.Sqrt(6f / (hiddenSize + hiddenSize));
            float kvLimit = MathF.Sqrt(6f / (hiddenSize + _kvDim));

            _wq = InitializeMatrix(hiddenSize, hiddenSize, qLimit, random);
            _wk = InitializeMatrix(hiddenSize, _kvDim, kvLimit, random);
            _wv = InitializeMatrix(hiddenSize, _kvDim, kvLimit, random);
            _wo = InitializeMatrix(hiddenSize, hiddenSize, qLimit, random);

            _wqGradients = new float[hiddenSize, hiddenSize];
            _wkGradients = new float[hiddenSize, _kvDim];
            _wvGradients = new float[hiddenSize, _kvDim];
            _woGradients = new float[hiddenSize, hiddenSize];

            _wqOptimizer = new AdamMatrixOptimizer(hiddenSize, hiddenSize);
            _wkOptimizer = new AdamMatrixOptimizer(hiddenSize, _kvDim);
            _wvOptimizer = new AdamMatrixOptimizer(hiddenSize, _kvDim);
            _woOptimizer = new AdamMatrixOptimizer(hiddenSize, hiddenSize);
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

            _lastInput = (float[,,])input.Clone();
            _lastPositionOffset = positionOffset;

            var q = ProjectHeads(input, _wq, _numAttentionHeads, batchSize, seqLen);
            var k = ProjectHeads(input, _wk, _numKeyValueHeads, batchSize, seqLen);
            var v = ProjectHeads(input, _wv, _numKeyValueHeads, batchSize, seqLen);

            if (_rotary != null)
            {
                q = _rotary.Apply(q, positionOffset);
                k = _rotary.Apply(k, positionOffset);
            }

            _lastQRotated = q;
            _lastKRotated = k;
            _lastV = v;

            var probabilities = new float[batchSize, _numAttentionHeads, seqLen, seqLen];
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
                            probabilities[b, qh, i, j] = p;

                            for (int d = 0; d < _headDim; d++)
                                contextConcat[b, i, qh * _headDim + d] += p * v[b, j, kvh, d];
                        }
                    }
                }
            }

            _lastProbabilities = probabilities;
            _lastContextConcat = contextConcat;

            return MatMulBatch(contextConcat, _wo, batchSize, seqLen, _hiddenSize, _hiddenSize);
        }

        private float[,,,] ProjectHeads(float[,,] input, float[,] weight, int numHeads, int batchSize, int seqLen)
        {
            int outDim = weight.GetLength(1);
            var flatOut = MatMulBatch(input, weight, batchSize, seqLen, _hiddenSize, outDim);

            var reshaped = new float[batchSize, seqLen, numHeads, _headDim];
            for (int b = 0; b < batchSize; b++)
                for (int s = 0; s < seqLen; s++)
                    for (int h = 0; h < numHeads; h++)
                        for (int d = 0; d < _headDim; d++)
                            reshaped[b, s, h, d] = flatOut[b, s, h * _headDim + d];

            return reshaped;
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

            var q = ProjectHeads(input, _wq, _numAttentionHeads, batchSize, newSeqLen);
            var k = ProjectHeads(input, _wk, _numKeyValueHeads, batchSize, newSeqLen);
            var v = ProjectHeads(input, _wv, _numKeyValueHeads, batchSize, newSeqLen);

            if (_rotary != null)
            {
                q = _rotary.Apply(q, positionOffset);
                k = _rotary.Apply(k, positionOffset);
            }

            cache.Append(k, v);

            var cachedKeys = cache.GetKeys();
            var cachedValues = cache.GetValues();

            var contextConcat = new float[batchSize, newSeqLen, _hiddenSize];
            float scale = 1f / MathF.Sqrt(_headDim);

            for (int b = 0; b < batchSize; b++)
            {
                for (int qh = 0; qh < _numAttentionHeads; qh++)
                {
                    int kvh = qh / _numGroups;

                    for (int i = 0; i < newSeqLen; i++)
                    {
                        int allowedLength = positionOffset + i + 1;
                        var scores = new float[allowedLength];
                        float maxScore = float.NegativeInfinity;

                        for (int j = 0; j < allowedLength; j++)
                        {
                            float dot = 0f;
                            for (int d = 0; d < _headDim; d++)
                                dot += q[b, i, qh, d] * cachedKeys[b, j, kvh, d];

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
                                contextConcat[b, i, qh * _headDim + d] += p * cachedValues[b, j, kvh, d];
                        }
                    }
                }
            }

            return MatMulBatch(contextConcat, _wo, batchSize, newSeqLen, _hiddenSize, _hiddenSize);
        }

        private static float[,,] MatMulBatch(float[,,] input, float[,] weight, int batchSize, int seqLen, int inDim, int outDim)
        {
            var output = new float[batchSize, seqLen, outDim];

            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    for (int o = 0; o < outDim; o++)
                    {
                        float sum = 0f;
                        for (int i = 0; i < inDim; i++)
                            sum += input[b, s, i] * weight[i, o];
                        output[b, s, o] = sum;
                    }
                }
            }

            return output;
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
                WqOptimizerState = _wqOptimizer.SaveState(),
                WkOptimizerState = _wkOptimizer.SaveState(),
                WvOptimizerState = _wvOptimizer.SaveState(),
                WoOptimizerState = _woOptimizer.SaveState()
            };
        }

        public static GQAAttention LoadState(GQAAttentionState state, RotaryEmbedding? rotary = null)
        {
            var attention = new GQAAttention(state.HiddenSize, state.NumAttentionHeads, state.NumKeyValueHeads, rotary);
            attention._wq = (float[,])state.Wq.Clone();
            attention._wk = (float[,])state.Wk.Clone();
            attention._wv = (float[,])state.Wv.Clone();
            attention._wo = (float[,])state.Wo.Clone();

            if (state.WqOptimizerState != null) attention._wqOptimizer.LoadStateInto(state.WqOptimizerState);
            if (state.WkOptimizerState != null) attention._wkOptimizer.LoadStateInto(state.WkOptimizerState);
            if (state.WvOptimizerState != null) attention._wvOptimizer.LoadStateInto(state.WvOptimizerState);
            if (state.WoOptimizerState != null) attention._woOptimizer.LoadStateInto(state.WoOptimizerState);

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
