using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class PositionalEncoding
    {
        private readonly int _maxSequenceLength;
        private readonly int _embeddingDim;
        private readonly Neuraval.Tensor.Tensor _encodings;

        public int MaxSequenceLength => _maxSequenceLength;
        public int EmbeddingDim => _embeddingDim;

        public PositionalEncoding(int maxSequenceLength, int embeddingDim)
        {
            _maxSequenceLength = maxSequenceLength;
            _embeddingDim = embeddingDim;
            _encodings = new Neuraval.Tensor.Tensor(new[] { maxSequenceLength, embeddingDim });

            ComputeEncodings();
        }

        private void ComputeEncodings()
        {
            for (int pos = 0; pos < _maxSequenceLength; pos++)
            {
                int rowOffset = pos * _embeddingDim;

                for (int i = 0; i < _embeddingDim; i++)
                {
                    // El ángulo se calcula en double a propósito (no en float): esto corre una
                    // sola vez por posición al construir el modelo, no en el hot path de
                    // entrenamiento/inferencia, así que no cuesta rendimiento, y Math.Pow con
                    // exponentes que dependen de la posición pierde menos precisión en double.
                    // Recién al guardar en el buffer del Tensor se castea hacia abajo.
                    double angle = pos / Math.Pow(10000.0, (2.0 * i) / _embeddingDim);

                    _encodings.Buffer[rowOffset + i] = i % 2 == 0
                        ? (float)Math.Sin(angle)
                        : (float)Math.Cos(angle);
                }
            }
        }

        private void ValidateDim(int embeddingDim)
        {
            if (embeddingDim != _embeddingDim)
            {
                throw new ArgumentException(
                    $"Embedding dimension {embeddingDim} does not match positional encoding dimension {_embeddingDim}");
            }
        }

        private void ValidateSequenceLength(int sequenceLength)
        {
            if (sequenceLength > _maxSequenceLength)
            {
                throw new ArgumentException(
                    $"Sequence length {sequenceLength} exceeds maximum {_maxSequenceLength}");
            }
        }

        private Neuraval.Tensor.Tensor SliceEncodings(int offset, int length)
        {
            var slice = new float[length * _embeddingDim];
            Array.Copy(_encodings.Buffer, offset * _embeddingDim, slice, 0, slice.Length);
            return new Neuraval.Tensor.Tensor(slice, new[] { length, _embeddingDim });
        }

        public float[] GetEncoding(int position)
        {
            if (position < 0 || position >= _maxSequenceLength)
            {
                throw new ArgumentOutOfRangeException(nameof(position),
                    $"Position must be between 0 and {_maxSequenceLength - 1}");
            }

            var encoding = new float[_embeddingDim];
            Array.Copy(_encodings.Buffer, position * _embeddingDim, encoding, 0, _embeddingDim);
            return encoding;
        }

        public float[,] GetEncodings(int sequenceLength)
        {
            ValidateSequenceLength(sequenceLength);

            var encodings = new float[sequenceLength, _embeddingDim];
            System.Buffer.BlockCopy(_encodings.Buffer, 0, encodings, 0, sequenceLength * _embeddingDim * sizeof(float));
            return encodings;
        }

        public float[,] AddToEmbeddings(float[,] embeddings)
        {
            int sequenceLength = embeddings.GetLength(0);
            int embeddingDim = embeddings.GetLength(1);

            ValidateDim(embeddingDim);
            ValidateSequenceLength(sequenceLength);

            var embeddingsTensor = Neuraval.Tensor.Tensor.FromArray2D(embeddings);
            var encodingsSlice = SliceEncodings(0, sequenceLength);
            var result = TensorOps.Add(embeddingsTensor, encodingsSlice);

            return result.ToArray2D();
        }

        public float[,] AddToEmbeddingsAtOffset(float[,] embeddings, int offset)
        {
            int sequenceLength = embeddings.GetLength(0);
            int embeddingDim = embeddings.GetLength(1);

            ValidateDim(embeddingDim);

            if (offset < 0)
            {
                throw new ArgumentException("offset no puede ser negativo");
            }

            if (offset + sequenceLength > _maxSequenceLength)
            {
                throw new ArgumentException(
                    $"La posición {offset + sequenceLength - 1} excede el máximo {_maxSequenceLength - 1}");
            }

            var embeddingsTensor = Neuraval.Tensor.Tensor.FromArray2D(embeddings);
            var encodingsSlice = SliceEncodings(offset, sequenceLength);
            var result = TensorOps.Add(embeddingsTensor, encodingsSlice);

            return result.ToArray2D();
        }

        public float[,,] AddToEmbeddingsBatch(float[,,] embeddingsBatch)
        {
            int batchSize = embeddingsBatch.GetLength(0);
            int sequenceLength = embeddingsBatch.GetLength(1);
            int embeddingDim = embeddingsBatch.GetLength(2);

            ValidateDim(embeddingDim);
            ValidateSequenceLength(sequenceLength);

            int rowElems = sequenceLength * embeddingDim;

            var embeddingsFlat = new float[batchSize * rowElems];
            System.Buffer.BlockCopy(embeddingsBatch, 0, embeddingsFlat, 0, embeddingsFlat.Length * sizeof(float));

            // Repetimos el bloque de encodings una vez por elemento del batch: son
            // memcpy contiguos (O(batchSize) copias), no un loop manual por elemento.
            var tiledEncodings = new float[batchSize * rowElems];
            for (int b = 0; b < batchSize; b++)
            {
                Array.Copy(_encodings.Buffer, 0, tiledEncodings, b * rowElems, rowElems);
            }

            var sumTensor = TensorOps.Add(
                new Neuraval.Tensor.Tensor(embeddingsFlat, new[] { batchSize * sequenceLength, embeddingDim }),
                new Neuraval.Tensor.Tensor(tiledEncodings, new[] { batchSize * sequenceLength, embeddingDim }));

            var result = new float[batchSize, sequenceLength, embeddingDim];
            System.Buffer.BlockCopy(sumTensor.Buffer, 0, result, 0, sumTensor.Buffer.Length * sizeof(float));
            return result;
        }

        public void AddInPlace(float[,] embeddings)
        {
            int sequenceLength = embeddings.GetLength(0);
            int embeddingDim = embeddings.GetLength(1);

            ValidateDim(embeddingDim);
            ValidateSequenceLength(sequenceLength);

            int count = sequenceLength * embeddingDim;
            var embeddingsFlat = new float[count];
            System.Buffer.BlockCopy(embeddings, 0, embeddingsFlat, 0, count * sizeof(float));

            var embeddingsTensor = new Neuraval.Tensor.Tensor(embeddingsFlat, new[] { sequenceLength, embeddingDim });
            var encodingsSlice = SliceEncodings(0, sequenceLength);

            TensorOps.AddInPlace(embeddingsTensor, encodingsSlice);

            System.Buffer.BlockCopy(embeddingsTensor.Buffer, 0, embeddings, 0, count * sizeof(float));
        }

        public float[,] GetAllEncodings()
        {
            var result = new float[_maxSequenceLength, _embeddingDim];
            System.Buffer.BlockCopy(_encodings.Buffer, 0, result, 0, _encodings.Buffer.Length * sizeof(float));
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
