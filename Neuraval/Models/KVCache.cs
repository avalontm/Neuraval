using System;
using System.Collections.Generic;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class KVCacheLayer
    {
        private readonly Neuraval.Tensor.Tensor _keys;
        private readonly Neuraval.Tensor.Tensor _values;
        private int _length;

        public int Capacity { get; }
        public int EmbeddingDim { get; }
        public int Length => _length;

        public KVCacheLayer(int capacity, int embeddingDim)
        {
            if (capacity <= 0)
                throw new ArgumentException("La capacidad del KVCache debe ser positiva");

            if (embeddingDim <= 0)
                throw new ArgumentException("EmbeddingDim debe ser positivo");

            Capacity = capacity;
            EmbeddingDim = embeddingDim;

            _keys = new Neuraval.Tensor.Tensor(new[] { capacity, embeddingDim });
            _values = new Neuraval.Tensor.Tensor(new[] { capacity, embeddingDim });
            _length = 0;
        }

        public void Append(float[,] newKeys, float[,] newValues)
        {
            int newCount = newKeys.GetLength(0);

            if (newKeys.GetLength(1) != EmbeddingDim || newValues.GetLength(1) != EmbeddingDim)
                throw new ArgumentException("Las dimensiones de keys/values no coinciden con el EmbeddingDim del cache");

            if (_length + newCount > Capacity)
                throw new InvalidOperationException($"El KVCache excede su capacidad ({Capacity} posiciones)");

            int byteCount = newCount * EmbeddingDim * sizeof(float);
            int destOffset = _length * EmbeddingDim * sizeof(float);

            Buffer.BlockCopy(newKeys, 0, _keys.Buffer, destOffset, byteCount);
            Buffer.BlockCopy(newValues, 0, _values.Buffer, destOffset, byteCount);

            _length += newCount;
        }

        public float[,] GetKeys() => ExtractRows(_keys);

        public float[,] GetValues() => ExtractRows(_values);

        private float[,] ExtractRows(Neuraval.Tensor.Tensor source)
        {
            var result = new float[_length, EmbeddingDim];
            Buffer.BlockCopy(source.Buffer, 0, result, 0, _length * EmbeddingDim * sizeof(float));
            return result;
        }

        public void Reset()
        {
            _length = 0;
        }
    }

    public class GenerationCache
    {
        public List<KVCacheLayer> Layers { get; }
        public int Length => Layers.Count > 0 ? Layers[0].Length : 0;
        public int Capacity { get; }

        public GenerationCache(int numLayers, int capacity, int embeddingDim)
        {
            if (numLayers <= 0)
                throw new ArgumentException("numLayers debe ser positivo");

            Capacity = capacity;
            Layers = new List<KVCacheLayer>(numLayers);

            for (int i = 0; i < numLayers; i++)
            {
                Layers.Add(new KVCacheLayer(capacity, embeddingDim));
            }
        }

        public void Reset()
        {
            foreach (var layer in Layers)
            {
                layer.Reset();
            }
        }
    }
}
