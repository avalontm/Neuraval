namespace Neuraval.Tensor.Backends.Cpu
{
    internal static class MatrixLayout
    {
        public static float[] Flatten(float[,] source)
        {
            int rows = source.GetLength(0);
            int cols = source.GetLength(1);
            var flat = new float[rows * cols];

            System.Buffer.BlockCopy(source, 0, flat, 0, flat.Length * sizeof(float));

            return flat;
        }

        public static float[] Transpose(float[] source, int rows, int cols)
        {
            var result = new float[rows * cols];

            for (int i = 0; i < rows; i++)
            {
                int rowOffset = i * cols;

                for (int j = 0; j < cols; j++)
                {
                    result[j * rows + i] = source[rowOffset + j];
                }
            }

            return result;
        }
    }
}
