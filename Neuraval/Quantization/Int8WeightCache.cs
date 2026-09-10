using System;

namespace Neuraval.Core.Quantization
{
    /// <summary>
    /// Cachea la versión cuantizada en INT8 de una matriz de pesos para
    /// inferencia en CPU, análogo a <c>CudaWeightCacheFp16</c> (Fase 4.3) pero
    /// sin depender de CUDA. Se recuantiza únicamente cuando alguien llama a
    /// <see cref="Invalidate"/> (típicamente después de <c>UpdateWeights</c>
    /// o de cargar un estado nuevo), no en cada Forward.
    /// </summary>
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
