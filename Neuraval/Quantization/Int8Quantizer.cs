using System;

namespace Neuraval.Core.Quantization
{
    public readonly struct QuantizedMatrix
    {
        public int Rows { get; }
        public int Cols { get; }

        public sbyte[] Data { get; }

        public float[] RowScales { get; }

        public QuantizedMatrix(int rows, int cols, sbyte[] data, float[] rowScales)
        {
            Rows = rows;
            Cols = cols;
            Data = data;
            RowScales = rowScales;
        }

        public long QuantizedByteSize => (long)Rows * Cols * sizeof(sbyte);

        public long OriginalByteSize => (long)Rows * Cols * sizeof(float);
    }

    public static class Int8Quantizer
    {
        private const sbyte MaxLevel = 127;

        public static QuantizedMatrix QuantizeRowSymmetric(float[,] matrix)
        {
            if (matrix == null) throw new ArgumentNullException(nameof(matrix));

            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var flat = new float[rows * cols];
            Buffer.BlockCopy(matrix, 0, flat, 0, flat.Length * sizeof(float));

            return QuantizeRowSymmetric(flat, rows, cols);
        }

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
