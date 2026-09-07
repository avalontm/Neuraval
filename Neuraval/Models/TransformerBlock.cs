using Neuraval.Core.Utils;

namespace Neuraval.Core.Models
{
    public class TransformerBlock
    {
        private readonly int _embeddingDim;
        private readonly int _numHeads;
        private readonly int _feedforwardDim;
        private readonly float _dropout;

        private readonly Random _dropoutRandom;

        private MultiHeadAttention _attention;
        private FeedForwardNetwork _feedforward;
        private LayerNormalization _norm1;
        private LayerNormalization _norm2;

        private float[,]? _lastAttentionDropoutMask;
        private float[,]? _lastFeedforwardDropoutMask;

        private float[,,]? _lastAttentionDropoutMaskBatch;
        private float[,,]? _lastFeedforwardDropoutMaskBatch;

        public int EmbeddingDim => _embeddingDim;
        public int NumHeads => _numHeads;

        public TransformerBlock(int embeddingDim, int numHeads, int feedforwardDim, float dropout = 0.1f, int seed = 42)
        {
            _embeddingDim = embeddingDim;
            _numHeads = numHeads;
            _feedforwardDim = feedforwardDim;
            _dropout = dropout;

            _attention = new MultiHeadAttention(embeddingDim, numHeads, seed);
            _feedforward = new FeedForwardNetwork(embeddingDim, feedforwardDim, seed + 1);
            _norm1 = new LayerNormalization(embeddingDim);
            _norm2 = new LayerNormalization(embeddingDim);
            _dropoutRandom = new Random(seed + 987654);
        }

        public void ZeroGradients()
        {
            _attention.ZeroGradients();
            _feedforward.ZeroGradients();
            _norm1.ZeroGradients();
            _norm2.ZeroGradients();
        }

        public void AverageGradients(int batchSize)
        {
            _attention.AverageGradients(batchSize);
            _feedforward.AverageGradients(batchSize);
            _norm1.AverageGradients(batchSize);
            _norm2.AverageGradients(batchSize);
        }

        public void ClipGradients(float maxNorm)
        {
            _attention.ClipGradients(maxNorm);
            _feedforward.ClipGradients(maxNorm);
            _norm1.ClipGradients(maxNorm);
            _norm2.ClipGradients(maxNorm);
        }

        public float[,] Forward(float[,] input, float[,]? mask = null, bool training = true)
        {
            int seqLen = input.GetLength(0);
            int embDim = input.GetLength(1);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input dimension {embDim} does not match expected {_embeddingDim}");
            }

            var norm1Output = _norm1.Forward(input);
            var attentionOutput = _attention.Forward(norm1Output, mask);

            if (training)
            {
                attentionOutput = ApplyDropout(attentionOutput, _dropout, out _lastAttentionDropoutMask);
            }
            else
            {
                _lastAttentionDropoutMask = null;
            }

            var residual1 = AddResidual(input, attentionOutput);
            var norm2Output = _norm2.Forward(residual1);
            var ffOutput = _feedforward.Forward(norm2Output);

            if (training)
            {
                ffOutput = ApplyDropout(ffOutput, _dropout, out _lastFeedforwardDropoutMask);
            }
            else
            {
                _lastFeedforwardDropoutMask = null;
            }

            var output = AddResidual(residual1, ffOutput);

            return output;
        }

        public float[,] Backward(float[,] gradOutput)
        {
            var gradFeedforwardOutput = ApplyDropoutBackward(gradOutput, _lastFeedforwardDropoutMask);
            var gradNorm2Output = _feedforward.Backward(gradFeedforwardOutput, 0.0f);
            var gradResidual1FromNorm2 = _norm2.Backward(gradNorm2Output, 0.0f);
            var gradResidual1 = AddResidual(gradOutput, gradResidual1FromNorm2);

            var gradAttentionOutput = ApplyDropoutBackward(gradResidual1, _lastAttentionDropoutMask);
            var gradNorm1Output = _attention.Backward(gradAttentionOutput, 0.0f);
            var gradInputFromNorm1 = _norm1.Backward(gradNorm1Output, 0.0f);
            var gradInput = AddResidual(gradResidual1, gradInputFromNorm1);

            return gradInput;
        }

        public float[,,] ForwardBatch(float[,,] inputBatch, float[,]? mask = null, bool training = true)
        {
            int embDim = inputBatch.GetLength(2);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input dimension {embDim} does not match expected {_embeddingDim}");
            }

            var norm1Output = _norm1.ForwardBatch(inputBatch);
            var attentionOutput = _attention.ForwardBatch(norm1Output, mask);

            if (training)
            {
                attentionOutput = ApplyDropoutBatch(attentionOutput, _dropout, out _lastAttentionDropoutMaskBatch);
            }
            else
            {
                _lastAttentionDropoutMaskBatch = null;
            }

            var residual1 = Matematicas.BatchMatrixAdd(inputBatch, attentionOutput);
            var norm2Output = _norm2.ForwardBatch(residual1);
            var ffOutput = _feedforward.ForwardBatch(norm2Output);

            if (training)
            {
                ffOutput = ApplyDropoutBatch(ffOutput, _dropout, out _lastFeedforwardDropoutMaskBatch);
            }
            else
            {
                _lastFeedforwardDropoutMaskBatch = null;
            }

            var output = Matematicas.BatchMatrixAdd(residual1, ffOutput);

            return output;
        }

        public float[,,] ForwardBatch(float[,,] inputBatch, float[,,]? maskBatch, bool training = true)
        {
            int embDim = inputBatch.GetLength(2);

            if (embDim != _embeddingDim)
            {
                throw new ArgumentException($"Input dimension {embDim} does not match expected {_embeddingDim}");
            }

            var norm1Output = _norm1.ForwardBatch(inputBatch);
            var attentionOutput = _attention.ForwardBatch(norm1Output, maskBatch);

            if (training)
            {
                attentionOutput = ApplyDropoutBatch(attentionOutput, _dropout, out _lastAttentionDropoutMaskBatch);
            }
            else
            {
                _lastAttentionDropoutMaskBatch = null;
            }

            var residual1 = Matematicas.BatchMatrixAdd(inputBatch, attentionOutput);
            var norm2Output = _norm2.ForwardBatch(residual1);
            var ffOutput = _feedforward.ForwardBatch(norm2Output);

            if (training)
            {
                ffOutput = ApplyDropoutBatch(ffOutput, _dropout, out _lastFeedforwardDropoutMaskBatch);
            }
            else
            {
                _lastFeedforwardDropoutMaskBatch = null;
            }

            var output = Matematicas.BatchMatrixAdd(residual1, ffOutput);

            return output;
        }

        public float[,,] BackwardBatch(float[,,] gradOutputBatch)
        {
            var gradFeedforwardOutput = ApplyDropoutBackwardBatch(gradOutputBatch, _lastFeedforwardDropoutMaskBatch);
            var gradNorm2Output = _feedforward.BackwardBatch(gradFeedforwardOutput, 0.0f);
            var gradResidual1FromNorm2 = _norm2.BackwardBatch(gradNorm2Output, 0.0f);
            var gradResidual1 = Matematicas.BatchMatrixAdd(gradOutputBatch, gradResidual1FromNorm2);

            var gradAttentionOutput = ApplyDropoutBackwardBatch(gradResidual1, _lastAttentionDropoutMaskBatch);
            var gradNorm1Output = _attention.BackwardBatch(gradAttentionOutput, 0.0f);
            var gradInputFromNorm1 = _norm1.BackwardBatch(gradNorm1Output, 0.0f);
            var gradInput = Matematicas.BatchMatrixAdd(gradResidual1, gradInputFromNorm1);

            return gradInput;
        }

        private float[,] AddResidual(float[,] input, float[,] residual)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = input[i, j] + residual[i, j];
                }
            }

            return result;
        }

        private float[,] ApplyDropout(float[,] input, float dropoutRate, out float[,]? mask)
        {
            if (dropoutRate <= 0)
            {
                mask = null;
                return input;
            }

            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            var result = new float[rows, cols];
            var maskMatrix = new float[rows, cols];

            float scale = 1.0f / (1.0f - dropoutRate);

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    if (_dropoutRandom.NextSingle() > dropoutRate)
                    {
                        result[i, j] = input[i, j] * scale;
                        maskMatrix[i, j] = scale;
                    }
                    else
                    {
                        result[i, j] = 0;
                        maskMatrix[i, j] = 0;
                    }
                }
            }

            mask = maskMatrix;
            return result;
        }

        private float[,] ApplyDropoutBackward(float[,] gradOutput, float[,]? mask)
        {
            if (mask == null)
            {
                return gradOutput;
            }

            int rows = gradOutput.GetLength(0);
            int cols = gradOutput.GetLength(1);
            var result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = gradOutput[i, j] * mask[i, j];
                }
            }

            return result;
        }

        private float[,,] ApplyDropoutBatch(float[,,] inputBatch, float dropoutRate, out float[,,]? maskBatch)
        {
            if (dropoutRate <= 0)
            {
                maskBatch = null;
                return inputBatch;
            }

            int batchSize = inputBatch.GetLength(0);
            int seqLen = inputBatch.GetLength(1);
            int dim = inputBatch.GetLength(2);
            var result = new float[batchSize, seqLen, dim];
            var maskMatrix = new float[batchSize, seqLen, dim];

            float scale = 1.0f / (1.0f - dropoutRate);

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        if (_dropoutRandom.NextSingle() > dropoutRate)
                        {
                            result[b, i, j] = inputBatch[b, i, j] * scale;
                            maskMatrix[b, i, j] = scale;
                        }
                        else
                        {
                            result[b, i, j] = 0;
                            maskMatrix[b, i, j] = 0;
                        }
                    }
                }
            }

            maskBatch = maskMatrix;
            return result;
        }

        private float[,,] ApplyDropoutBackwardBatch(float[,,] gradOutputBatch, float[,,]? maskBatch)
        {
            if (maskBatch == null)
            {
                return gradOutputBatch;
            }

            int batchSize = gradOutputBatch.GetLength(0);
            int seqLen = gradOutputBatch.GetLength(1);
            int dim = gradOutputBatch.GetLength(2);
            var result = new float[batchSize, seqLen, dim];

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        result[b, i, j] = gradOutputBatch[b, i, j] * maskBatch[b, i, j];
                    }
                }
            }

            return result;
        }

        public void UpdateWeights(float learningRate)
        {
            _attention.UpdateWeights(learningRate);
            _feedforward.UpdateWeights(learningRate);
            _norm1.UpdateWeights(learningRate);
            _norm2.UpdateWeights(learningRate);
        }

        public void ResetGradients()
        {
            _attention.ResetGradients();
            _feedforward.ResetGradients();
            _norm1.ResetGradients();
            _norm2.ResetGradients();
        }

        public TransformerBlockState SaveState()
        {
            return new TransformerBlockState
            {
                EmbeddingDim = _embeddingDim,
                NumHeads = _numHeads,
                FeedforwardDim = _feedforwardDim,
                Dropout = _dropout,
                AttentionState = _attention.SaveState(),
                FeedforwardState = _feedforward.SaveState(),
                Norm1State = _norm1.SaveState(),
                Norm2State = _norm2.SaveState()
            };
        }

        public static TransformerBlock LoadState(TransformerBlockState state)
        {
            var block = new TransformerBlock(
                state.EmbeddingDim,
                state.NumHeads,
                state.FeedforwardDim,
                state.Dropout);

            block._attention = MultiHeadAttention.LoadState(state.AttentionState);
            block._feedforward = FeedForwardNetwork.LoadState(state.FeedforwardState);
            block._norm1 = LayerNormalization.LoadState(state.Norm1State);
            block._norm2 = LayerNormalization.LoadState(state.Norm2State);

            return block;
        }
    }

    public class TransformerBlockState
    {
        public int EmbeddingDim { get; set; }
        public int NumHeads { get; set; }
        public int FeedforwardDim { get; set; }
        public float Dropout { get; set; }
        public MultiHeadAttentionState AttentionState { get; set; }
        public FeedForwardNetworkState FeedforwardState { get; set; }
        public LayerNormalizationState Norm1State { get; set; }
        public LayerNormalizationState Norm2State { get; set; }

        public TransformerBlockState()
        {
            AttentionState = new MultiHeadAttentionState();
            FeedforwardState = new FeedForwardNetworkState();
            Norm1State = new LayerNormalizationState();
            Norm2State = new LayerNormalizationState();
        }
    }
}