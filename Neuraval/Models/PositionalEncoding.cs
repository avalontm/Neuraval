namespace Neuraval.Core.Models
{
    public class PositionalEncoding
    {
        private readonly int _maxSequenceLength;
        private readonly int _embeddingDim;
        private readonly float[,] _encodings;

        public int MaxSequenceLength => _maxSequenceLength;
        public int EmbeddingDim => _embeddingDim;

        public PositionalEncoding(int maxSequenceLength, int embeddingDim)
        {
            _maxSequenceLength = maxSequenceLength;
            _embeddingDim = embeddingDim;
            _encodings = new float[maxSequenceLength, embeddingDim];

            ComputeEncodings();
        }

        private void ComputeEncodings()
        {
            for (int pos = 0; pos < _maxSequenceLength; pos++)
            {
                for (int i = 0; i < _embeddingDim; i++)
                {
                    // El ángulo se calcula en double a propósito (no en float): esto corre una
                    // sola vez por posición al construir el modelo, no en el hot path de
                    // entrenamiento/inferencia, así que no cuesta rendimiento, y Math.Pow con
                    // exponentes que dependen de la posición pierde menos precisión en double.
                    // Recién al guardar en _encodings (float[,]) se castea hacia abajo.
                    double angle = pos / Math.Pow(10000.0, (2.0 * i) / _embeddingDim);

                    if (i % 2 == 0)
                    {
                        _encodings[pos, i] = (float)Math.Sin(angle);
                    }
                    else
                    {
                        _encodings[pos, i] = (float)Math.Cos(angle);
                    }
                }
            }
        }

        public float[] GetEncoding(int position)
        {
            if (position < 0 || position >= _maxSequenceLength)
            {
                throw new ArgumentOutOfRangeException(nameof(position),
                    $"Position must be between 0 and {_maxSequenceLength - 1}");
            }

            var encoding = new float[_embeddingDim];
            for (int i = 0; i < _embeddingDim; i++)
            {
                encoding[i] = _encodings[position, i];
            }

            return encoding;
        }

        public float[,] GetEncodings(int sequenceLength)
        {
            if (sequenceLength > _maxSequenceLength)
            {
                throw new ArgumentException(
                    $"Sequence length {sequenceLength} exceeds maximum {_maxSequenceLength}");
            }

            var encodings = new float[sequenceLength, _embeddingDim];

            for (int pos = 0; pos < sequenceLength; pos++)
            {
                for (int i = 0; i < _embeddingDim; i++)
                {
                    encodings[pos, i] = _encodings[pos, i];
                }
            }

            return encodings;
        }

        public float[,] AddToEmbeddings(float[,] embeddings)
        {
            int sequenceLength = embeddings.GetLength(0);
            int embeddingDim = embeddings.GetLength(1);

            if (embeddingDim != _embeddingDim)
            {
                throw new ArgumentException(
                    $"Embedding dimension {embeddingDim} does not match positional encoding dimension {_embeddingDim}");
            }

            if (sequenceLength > _maxSequenceLength)
            {
                throw new ArgumentException(
                    $"Sequence length {sequenceLength} exceeds maximum {_maxSequenceLength}");
            }

            var result = new float[sequenceLength, embeddingDim];

            for (int pos = 0; pos < sequenceLength; pos++)
            {
                for (int dim = 0; dim < embeddingDim; dim++)
                {
                    result[pos, dim] = embeddings[pos, dim] + _encodings[pos, dim];
                }
            }

            return result;
        }

        public float[,,] AddToEmbeddingsBatch(float[,,] embeddingsBatch)
        {
            int batchSize = embeddingsBatch.GetLength(0);
            int sequenceLength = embeddingsBatch.GetLength(1);
            int embeddingDim = embeddingsBatch.GetLength(2);

            if (embeddingDim != _embeddingDim)
            {
                throw new ArgumentException(
                    $"Embedding dimension {embeddingDim} does not match positional encoding dimension {_embeddingDim}");
            }

            if (sequenceLength > _maxSequenceLength)
            {
                throw new ArgumentException(
                    $"Sequence length {sequenceLength} exceeds maximum {_maxSequenceLength}");
            }

            var result = new float[batchSize, sequenceLength, embeddingDim];

            for (int b = 0; b < batchSize; b++)
            {
                for (int pos = 0; pos < sequenceLength; pos++)
                {
                    for (int dim = 0; dim < embeddingDim; dim++)
                    {
                        result[b, pos, dim] = embeddingsBatch[b, pos, dim] + _encodings[pos, dim];
                    }
                }
            }

            return result;
        }

        public void AddInPlace(float[,] embeddings)
        {
            int sequenceLength = embeddings.GetLength(0);
            int embeddingDim = embeddings.GetLength(1);

            if (embeddingDim != _embeddingDim)
            {
                throw new ArgumentException(
                    $"Embedding dimension mismatch: expected {_embeddingDim}, got {embeddingDim}");
            }

            if (sequenceLength > _maxSequenceLength)
            {
                throw new ArgumentException(
                    $"Sequence length {sequenceLength} exceeds maximum {_maxSequenceLength}");
            }

            for (int pos = 0; pos < sequenceLength; pos++)
            {
                for (int dim = 0; dim < embeddingDim; dim++)
                {
                    embeddings[pos, dim] += _encodings[pos, dim];
                }
            }
        }

        public float[,] GetAllEncodings()
        {
            var result = new float[_maxSequenceLength, _embeddingDim];
            Array.Copy(_encodings, result, _encodings.Length);
            return result;
        }

        public PositionalEncodingState SaveState()
        {
            return new PositionalEncodingState
            {
                MaxSequenceLength = _maxSequenceLength,
                EmbeddingDim = _embeddingDim
            };
        }

        public static PositionalEncoding LoadState(PositionalEncodingState state)
        {
            return new PositionalEncoding(state.MaxSequenceLength, state.EmbeddingDim);
        }
    }

    public class PositionalEncodingState
    {
        public int MaxSequenceLength { get; set; }
        public int EmbeddingDim { get; set; }
    }
}