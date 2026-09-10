using System;
using Neuraval.Core.Utils;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class EmbeddingLayer
    {
        private readonly float[,] _embeddings;
        private readonly int _vocabSize;
        private readonly int _embeddingDim;
        private readonly Random _random;
        private float[,] _gradients;
        private float[,] _accumulatedGradients;
        private AdamMatrixOptimizer _optimizer = null!;

        public int VocabSize => _vocabSize;
        public int EmbeddingDim => _embeddingDim;
        public float[,] EmbeddingsRef => _embeddings;

        public EmbeddingLayer(int vocabSize, int embeddingDim, int seed = 42)
        {
            _vocabSize = vocabSize;
            _embeddingDim = embeddingDim;
            _random = new Random(seed);

            _embeddings = new float[vocabSize, embeddingDim];
            _gradients = new float[vocabSize, embeddingDim];
            _accumulatedGradients = new float[vocabSize, embeddingDim];
            _optimizer = new AdamMatrixOptimizer(vocabSize, embeddingDim);

            InitializeXavier();
        }

        private void InitializeXavier()
        {
            float limit = MathF.Sqrt(6.0f / (_vocabSize + _embeddingDim));

            for (int i = 0; i < _vocabSize; i++)
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    _embeddings[i, j] = (_random.NextSingle() * 2f - 1f) * limit;
                }
            }
        }

        public void ZeroGradients()
        {
            Array.Clear(_accumulatedGradients, 0, _accumulatedGradients.Length);
        }

        public void AverageGradients(int batchSize)
        {
            if (batchSize <= 0)
                throw new ArgumentException("Batch size must be positive");

            float scale = 1.0f / batchSize;

            var accumulatedTensor = Neuraval.Tensor.Tensor.FromArray2D(_accumulatedGradients);
            _gradients = TensorOps.Scale(accumulatedTensor, scale).ToArray2D();
        }

        public float SumSquaredGradients()
        {
            var gradientsTensor = Neuraval.Tensor.Tensor.FromArray2D(_gradients);
            var squared = TensorOps.Multiply(gradientsTensor, gradientsTensor);
            return TensorOps.Sum(squared);
        }

        public void ScaleGradients(float scale)
        {
            var gradientsTensor = Neuraval.Tensor.Tensor.FromArray2D(_gradients);
            _gradients = TensorOps.Scale(gradientsTensor, scale).ToArray2D();
        }

        public float[] GetEmbedding(int tokenIndex)
        {
            if (tokenIndex < 0 || tokenIndex >= _vocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokenIndex),
                    $"Token index must be between 0 and {_vocabSize - 1}");
            }

            var embedding = new float[_embeddingDim];
            for (int i = 0; i < _embeddingDim; i++)
            {
                embedding[i] = _embeddings[tokenIndex, i];
            }

            return embedding;
        }

        public float[,] GetEmbeddings(int[] tokenIndices)
        {
            int sequenceLength = tokenIndices.Length;
            var embeddings = new float[sequenceLength, _embeddingDim];

            for (int i = 0; i < sequenceLength; i++)
            {
                var embedding = GetEmbedding(tokenIndices[i]);
                for (int j = 0; j < _embeddingDim; j++)
                {
                    embeddings[i, j] = embedding[j];
                }
            }

            return embeddings;
        }

        public float[,,] GetEmbeddingsBatch(int[,] tokenIndicesBatch)
        {
            int batchSize = tokenIndicesBatch.GetLength(0);
            int sequenceLength = tokenIndicesBatch.GetLength(1);
            var embeddings = new float[batchSize, sequenceLength, _embeddingDim];

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < sequenceLength; i++)
                {
                    var embedding = GetEmbedding(tokenIndicesBatch[b, i]);
                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        embeddings[b, i, j] = embedding[j];
                    }
                }
            }

            return embeddings;
        }

        public void BackwardBatch(int[,] tokenIndicesBatch, float[,,] gradOutputBatch)
        {
            int batchSize = tokenIndicesBatch.GetLength(0);
            int sequenceLength = tokenIndicesBatch.GetLength(1);

            if (gradOutputBatch.GetLength(0) != batchSize ||
                gradOutputBatch.GetLength(1) != sequenceLength ||
                gradOutputBatch.GetLength(2) != _embeddingDim)
            {
                throw new ArgumentException(
                    $"gradOutputBatch debe ser de tamaño {batchSize}x{sequenceLength}x{_embeddingDim}, es " +
                    $"{gradOutputBatch.GetLength(0)}x{gradOutputBatch.GetLength(1)}x{gradOutputBatch.GetLength(2)}");
            }

            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < sequenceLength; i++)
                {
                    var gradientRow = new float[_embeddingDim];
                    for (int j = 0; j < _embeddingDim; j++)
                    {
                        gradientRow[j] = gradOutputBatch[b, i, j];
                    }

                    AccumulateGradients(tokenIndicesBatch[b, i], gradientRow);
                }
            }
        }

        public void AccumulateGradients(int tokenIndex, float[] gradient)
        {
            if (tokenIndex < 0 || tokenIndex >= _vocabSize)
            {
                return;
            }

            if (gradient.Length != _embeddingDim)
            {
                throw new ArgumentException($"Gradient must have length {_embeddingDim}");
            }

            for (int i = 0; i < _embeddingDim; i++)
            {
                _accumulatedGradients[tokenIndex, i] += gradient[i];
            }
        }

        public void AccumulateGradientAt(int vocabIndex, int dimIndex, float value)
        {
            _accumulatedGradients[vocabIndex, dimIndex] += value;
        }

        public void Backward(int[] tokenIndices, float[,] gradOutput)
        {
            int sequenceLength = tokenIndices.Length;

            if (gradOutput.GetLength(0) != sequenceLength || gradOutput.GetLength(1) != _embeddingDim)
            {
                throw new ArgumentException(
                    $"gradOutput debe ser de tamaño {sequenceLength}x{_embeddingDim}, es {gradOutput.GetLength(0)}x{gradOutput.GetLength(1)}");
            }

            for (int i = 0; i < sequenceLength; i++)
            {
                var gradientRow = new float[_embeddingDim];
                for (int j = 0; j < _embeddingDim; j++)
                {
                    gradientRow[j] = gradOutput[i, j];
                }

                AccumulateGradients(tokenIndices[i], gradientRow);
            }
        }

        public void UpdateWeights(float learningRate)
        {
            _optimizer.Update(_embeddings, _gradients, learningRate);
            Array.Clear(_gradients, 0, _gradients.Length);
        }

        public void ResetGradients()
        {
            Array.Clear(_gradients, 0, _gradients.Length);
            Array.Clear(_accumulatedGradients, 0, _accumulatedGradients.Length);
        }

        public EmbeddingLayerState SaveState()
        {
            var state = new EmbeddingLayerState
            {
                VocabSize = _vocabSize,
                EmbeddingDim = _embeddingDim,
                Embeddings = new float[_vocabSize * _embeddingDim],
                OptimizerState = _optimizer.SaveState()
            };

            int index = 0;
            for (int i = 0; i < _vocabSize; i++)
            {
                for (int j = 0; j < _embeddingDim; j++)
                {
                    state.Embeddings[index++] = _embeddings[i, j];
                }
            }

            return state;
        }

        public static EmbeddingLayer LoadState(EmbeddingLayerState state)
        {
            var layer = new EmbeddingLayer(state.VocabSize, state.EmbeddingDim);

            int index = 0;
            for (int i = 0; i < state.VocabSize; i++)
            {
                for (int j = 0; j < state.EmbeddingDim; j++)
                {
                    layer._embeddings[i, j] = state.Embeddings[index++];
                }
            }

            if (state.OptimizerState != null)
            {
                layer._optimizer.LoadStateInto(state.OptimizerState);
            }

            return layer;
        }

        public float[,] GetAllEmbeddings()
        {
            var result = new float[_vocabSize, _embeddingDim];
            Array.Copy(_embeddings, result, _embeddings.Length);
            return result;
        }

        public void SetEmbedding(int tokenIndex, float[] embedding)
        {
            if (tokenIndex < 0 || tokenIndex >= _vocabSize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokenIndex));
            }

            if (embedding.Length != _embeddingDim)
            {
                throw new ArgumentException($"Embedding must have length {_embeddingDim}");
            }

            for (int i = 0; i < _embeddingDim; i++)
            {
                _embeddings[tokenIndex, i] = embedding[i];
            }
        }
    }

    public class EmbeddingLayerState
    {
        public int VocabSize { get; set; }
        public int EmbeddingDim { get; set; }
        public float[] Embeddings { get; set; }
        public AdamMatrixOptimizerState? OptimizerState { get; set; }

        public EmbeddingLayerState()
        {
            Embeddings = Array.Empty<float>();
        }
    }
}
