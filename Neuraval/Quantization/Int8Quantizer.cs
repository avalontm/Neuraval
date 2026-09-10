using System;

namespace Neuraval.Core.Quantization
{
    /// <summary>
    /// Resultado de cuantizar una matriz de pesos a INT8: los valores originales
    /// (float32) quedan aproximados como <c>valor ≈ Data[i,j] * RowScales[i]</c>.
    /// Se usa una escala independiente por fila (cuantización "per-row"/"per-channel")
    /// en vez de una única escala global, porque las filas de una matriz de pesos
    /// entrenada suelen tener rangos muy distintos entre sí; una sola escala global
    /// desperdiciaría resolución en las filas con valores chicos.
    /// </summary>
    public readonly struct QuantizedMatrix
    {
        public int Rows { get; }
        public int Cols { get; }

        /// <summary>Valores cuantizados, row-major, tamaño Rows*Cols.</summary>
        public sbyte[] Data { get; }

        /// <summary>Escala por fila (tamaño Rows). Escala = maxAbs(fila) / 127.</summary>
        public float[] RowScales { get; }

        public QuantizedMatrix(int rows, int cols, sbyte[] data, float[] rowScales)
        {
            Rows = rows;
            Cols = cols;
            Data = data;
            RowScales = rowScales;
        }

        /// <summary>Bytes ocupados por los datos cuantizados (sin contar las escalas).</summary>
        public long QuantizedByteSize => (long)Rows * Cols * sizeof(sbyte);

        /// <summary>Bytes que ocuparía la misma matriz sin cuantizar, como float32.</summary>
        public long OriginalByteSize => (long)Rows * Cols * sizeof(float);
    }

    /// <summary>
    /// Cuantización/decuantización INT8 simétrica por fila para pesos ya
    /// entrenados. Pensada exclusivamente para inferencia post-entrenamiento
    /// (Fase 5.4): no se usa en ningún punto del path de entrenamiento
    /// (Forward/Backward/ForwardBatch/BackwardBatch), que sigue intacto.
    ///
    /// Esquema: por cada fila de la matriz se calcula
    /// <c>scale = max(|fila|) / 127</c> y cada valor se guarda como
    /// <c>round(valor / scale)</c>, saturado a [-127, 127]. Es simétrico
    /// (sin zero-point) porque los pesos de una red entrenada suelen estar
    /// centrados en 0, a diferencia de activaciones o imágenes.
    /// </summary>
    public static class Int8Quantizer
    {
        private const sbyte MaxLevel = 127;

        /// <summary>
        /// Sobrecarga de conveniencia para matrices representadas como
        /// <c>float[,]</c> (la forma en la que Neuraval guarda los pesos en
        /// memoria durante inferencia, a diferencia del arreglo plano que usa
        /// la serialización).
        /// </summary>
        public static QuantizedMatrix QuantizeRowSymmetric(float[,] matrix)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var flat = new float[rows * cols];
            Buffer.BlockCopy(matrix, 0, flat, 0, flat.Length * sizeof(float));

            return QuantizeRowSymmetric(flat, rows, cols);
        }

        /// <summary>
        /// Cuantiza una matriz almacenada como arreglo plano row-major
        /// (<paramref name="rows"/> x <paramref name="cols"/>) a INT8 con una
        /// escala por fila.
        /// </summary>
        public static QuantizedMatrix QuantizeRowSymmetric(float[] flatRowMajor, int rows, int cols)
        {
            if (flatRowMajor == null) throw new ArgumentNullException(nameof(flatRowMajor));
            if (rows < 0 || cols < 0) throw new ArgumentOutOfRangeException(nameof(rows), "Dimensiones negativas.");
            if ((long)rows * cols != flatRowMajor.Length)
            {
                throw new ArgumentException(
                    $"El arreglo tiene {flatRowMajor.Length} elementos pero rows*cols = {(long)rows * cols}.");
            }

            var data = new sbyte[flatRowMajor.Length];
            var scales = new float[rows];

            for (int r = 0; r < rows; r++)
            {
                int rowOffset = r * cols;

                float maxAbs = 0f;
                for (int c = 0; c < cols; c++)
                {
                    float abs = MathF.Abs(flatRowMajor[rowOffset + c]);
                    if (abs > maxAbs) maxAbs = abs;
                }

                // Fila de puros ceros (o vacía): escala 1 para no dividir por cero;
                // todos los valores cuantizados quedan en 0 de todas formas.
                float scale = maxAbs > 0f ? maxAbs / MaxLevel : 1f;
                scales[r] = scale;

                if (maxAbs == 0f) continue;

                float invScale = 1f / scale;
                for (int c = 0; c < cols; c++)
                {
                    int idx = rowOffset + c;
                    int quantized = (int)MathF.Round(flatRowMajor[idx] * invScale, MidpointRounding.AwayFromZero);
                    if (quantized > MaxLevel) quantized = MaxLevel;
                    if (quantized < -MaxLevel) quantized = -MaxLevel;
                    data[idx] = (sbyte)quantized;
                }
            }

            return new QuantizedMatrix(rows, cols, data, scales);
        }

        /// <summary>
        /// Reconstruye la matriz aproximada como arreglo plano row-major de floats.
        /// </summary>
        public static float[] Dequantize(in QuantizedMatrix quantized)
        {
            var result = new float[quantized.Data.Length];
            int cols = quantized.Cols;

            for (int r = 0; r < quantized.Rows; r++)
            {
                float scale = quantized.RowScales[r];
                int rowOffset = r * cols;
                for (int c = 0; c < cols; c++)
                {
                    int idx = rowOffset + c;
                    result[idx] = quantized.Data[idx] * scale;
                }
            }

            return result;
        }

        /// <summary>
        /// Calcula el error de cuantización (máximo y promedio, en valor
        /// absoluto) entre los pesos originales y su versión decuantizada.
        /// Es solo diagnóstico — pensado para imprimirse al cuantizar un
        /// modelo y confirmar que la pérdida de precisión es razonable.
        /// </summary>
        public static (float MaxAbsError, float MeanAbsError) ComputeQuantizationError(
            float[] original, float[] dequantized)
        {
            if (original.Length != dequantized.Length)
            {
                throw new ArgumentException("Los arreglos original y decuantizado deben tener el mismo tamaño.");
            }

            if (original.Length == 0) return (0f, 0f);

            float maxAbsError = 0f;
            double sumAbsError = 0;

            for (int i = 0; i < original.Length; i++)
            {
                float absError = MathF.Abs(original[i] - dequantized[i]);
                if (absError > maxAbsError) maxAbsError = absError;
                sumAbsError += absError;
            }

            return (maxAbsError, (float)(sumAbsError / original.Length));
        }
    }
}
