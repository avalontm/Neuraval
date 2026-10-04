using System.Threading.Tasks;
using Neuraval.Core.Utils;
using Neuraval.Tensor;

namespace Neuraval.Core.Quantization
{
    public static class Int8MatMul
    {
        public static Neuraval.Tensor.Tensor MatMulCachedB(Neuraval.Tensor.Tensor input, float[,] weights, Int8WeightCache cache)
        {
            int m = input.Shape[0];
            int k = input.Shape[1];
            int n = weights.GetLength(1);

            var quantized = cache.GetOrQuantize(weights);
            var inputBuffer = input.Buffer;
            var rowScales = quantized.RowScales;
            var data = quantized.Data;

            var result = new float[m * n];

            Parallel.For(0, m, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, row =>
            {
                var scaledRow = new float[k];
                int inputRowOffset = row * k;
                for (int kk = 0; kk < k; kk++)
                {
                    scaledRow[kk] = inputBuffer[inputRowOffset + kk] * rowScales[kk];
                }

                int resultRowOffset = row * n;
                for (int col = 0; col < n; col++)
                {
                    float sum = 0f;
                    for (int kk = 0; kk < k; kk++)
                    {
                        sum += scaledRow[kk] * data[kk * n + col];
                    }
                    result[resultRowOffset + col] = sum;
                }
            });

            return new Neuraval.Tensor.Tensor(result, new[] { m, n }, input.Device, input.DType);
        }
    }
}
