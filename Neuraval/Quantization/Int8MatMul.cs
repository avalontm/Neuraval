using System.Threading.Tasks;
using Neuraval.Core.Utils;
using Neuraval.Tensor;

namespace Neuraval.Core.Quantization
{
    /// <summary>
    /// Multiplicación de matrices <c>input [m,k] · weights [k,n] -> [m,n]</c>
    /// usando pesos cuantizados en INT8 (Fase 5.4, solo inferencia).
    ///
    /// Truco clave del esquema "per-row" elegido en <see cref="Int8Quantizer"/>:
    /// como cada fila <c>k</c> de la matriz de pesos comparte una única
    /// escala, esa escala se puede aplicar sobre <c>input[m,k]</c> ANTES del
    /// producto punto (<c>input[m,k] * scale[k]</c>), y lo que queda adentro
    /// del bucle más caliente es un producto entre un float ya escalado y un
    /// entero de 8 bits — sin tener que decuantizar la matriz de pesos
    /// completa a float en memoria en ningún momento.
    ///
    /// Esta es una implementación escalar paralelizada por filas (igual que
    /// el resto de Neuraval usa <c>Parallel.For</c> para atención y demás).
    /// No usa SIMD/intrínsecos todavía: el beneficio garantizado es la huella
    /// de memoria de los pesos (4x más chica que float32), no necesariamente
    /// velocidad — para eso conviene medir con el comando
    /// <c>--int8-benchmark</c> antes de habilitar <see cref="Int8InferenceSettings"/>
    /// por defecto en un entorno productivo.
    /// </summary>
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
