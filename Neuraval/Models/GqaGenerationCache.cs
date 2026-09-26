using System;
using System.Collections.Generic;

namespace Neuraval.Core.Models
{
    public sealed class GqaKeyValueCacheLayer
    {
        private readonly float[,,,] _keys;
        private readonly float[,,,] _values;
        private int _length;

        public int BatchSize { get; }
        public int Capacity { get; }
        public int NumKeyValueHeads { get; }
        public int HeadDim { get; }
        public int Length => _length;

        public GqaKeyValueCacheLayer(int batchSize, int capacity, int numKeyValueHeads, int headDim)
        {
            if (batchSize <= 0)
                throw new ArgumentException("batchSize debe ser positivo");

            if (capacity <= 0)
                throw new ArgumentException("capacity debe ser positivo");

            if (numKeyValueHeads <= 0)
                throw new ArgumentException("numKeyValueHeads debe ser positivo");

            if (headDim <= 0)
                throw new ArgumentException("headDim debe ser positivo");

            BatchSize = batchSize;
            Capacity = capacity;
            NumKeyValueHeads = numKeyValueHeads;
            HeadDim = headDim;

            _keys = new float[batchSize, capacity, numKeyValueHeads, headDim];
            _values = new float[batchSize, capacity, numKeyValueHeads, headDim];
            _length = 0;
        }

        public void Append(float[,,,] newKeys, float[,,,] newValues)
        {
            if (newKeys == null)
                throw new ArgumentNullException(nameof(newKeys));

            if (newValues == null)
                throw new ArgumentNullException(nameof(newValues));

            int batchSize = newKeys.GetLength(0);
            int newCount = newKeys.GetLength(1);

            if (batchSize != BatchSize)
                throw new ArgumentException($"batchSize ({batchSize}) no coincide con el cache ({BatchSize})");

            if (newKeys.GetLength(2) != NumKeyValueHeads || newValues.GetLength(2) != NumKeyValueHeads)
                throw new ArgumentException("numKeyValueHeads no coincide con el cache");

            if (newKeys.GetLength(3) != HeadDim || newValues.GetLength(3) != HeadDim)
                throw new ArgumentException("headDim no coincide con el cache");

            if (_length + newCount > Capacity)
                throw new InvalidOperationException($"El cache excede su capacidad ({Capacity} posiciones)");

            for (int b = 0; b < BatchSize; b++)
                for (int s = 0; s < newCount; s++)
                    for (int h = 0; h < NumKeyValueHeads; h++)
                        for (int d = 0; d < HeadDim; d++)
                        {
                            _keys[b, _length + s, h, d] = newKeys[b, s, h, d];
                            _values[b, _length + s, h, d] = newValues[b, s, h, d];
                        }

            _length += newCount;
        }

        public float[,,,] GetKeys() => ExtractRows(_keys);

        public float[,,,] GetValues() => ExtractRows(_values);

        private float[,,,] ExtractRows(float[,,,] source)
        {
            var result = new float[BatchSize, _length, NumKeyValueHeads, HeadDim];

            for (int b = 0; b < BatchSize; b++)
                for (int s = 0; s < _length; s++)
                    for (int h = 0; h < NumKeyValueHeads; h++)
                        for (int d = 0; d < HeadDim; d++)
                            result[b, s, h, d] = source[b, s, h, d];

            return result;
        }

        public long EstimatedMemoryBytes =>
            2L * BatchSize * Capacity * NumKeyValueHeads * HeadDim * sizeof(float);

        public void Reset()
        {
            _length = 0;
        }
    }

    public sealed class GqaGenerationCache
    {
        public IReadOnlyList<GqaKeyValueCacheLayer> Layers { get; }
        public int Length => Layers.Count > 0 ? Layers[0].Length : 0;
        public int Capacity { get; }
        public int BatchSize { get; }

        public GqaGenerationCache(int numLayers, int batchSize, int capacity, int numKeyValueHeads, int headDim)
        {
            if (numLayers <= 0)
                throw new ArgumentException("numLayers debe ser positivo");

            Capacity = capacity;
            BatchSize = batchSize;

            var layers = new List<GqaKeyValueCacheLayer>(numLayers);
            for (int i = 0; i < numLayers; i++)
                layers.Add(new GqaKeyValueCacheLayer(batchSize, capacity, numKeyValueHeads, headDim));

            Layers = layers;
        }

        public long EstimatedMemoryBytes
        {
            get
            {
                long total = 0;
                foreach (var layer in Layers)
                    total += layer.EstimatedMemoryBytes;
                return total;
            }
        }

        public void Reset()
        {
            foreach (var layer in Layers)
                layer.Reset();
        }
    }
}
