using System;

namespace Neuraval.Core.Quantization
{
    public sealed class Int8WeightCache
    {
        private readonly int _rows;
        private readonly int _cols;
        private QuantizedMatrix? _cached;
        private bool _dirty;

        public Int8WeightCache(int rows, int cols)
        {
            _rows = rows;
            _cols = cols;
            _dirty = true;
        }

        public void Invalidate()
        {
            _dirty = true;
        }

        public QuantizedMatrix GetOrQuantize(float[,] weights)
        {
            if (weights.GetLength(0) != _rows || weights.GetLength(1) != _cols)
            {
                throw new ArgumentException("La forma de la matriz de pesos no coincide con la del cache.");
            }

            if (_cached == null || _dirty)
            {
                _cached = Int8Quantizer.QuantizeRowSymmetric(weights);
                _dirty = false;
            }

            return _cached.Value;
        }
    }
}
