using System;
using System.Numerics;
using System.Threading.Tasks;
using Neuraval.Cuda;

namespace Neuraval.Core.Utils
{
    public static class Matematicas
    {
        private static int _numThreads = Environment.ProcessorCount;
        private static bool _gpuEnabled = false;

        public static void SetNumThreads(int numThreads)
        {
            _numThreads = numThreads > 0 ? numThreads : Environment.ProcessorCount;
        }

        public static int GetNumThreads()
        {
            return _numThreads;
        }

        public static bool GpuEnabled => _gpuEnabled;

        public static bool IsGpuAvailable()
        {
            try
            {
                return CudaDevice.IsAvailable();
            }
            catch
            {
                return false;
            }
        }

        public static bool TryEnableGpu(int deviceId = 0)
        {
            try
            {
                if (!CudaDevice.IsAvailable())
                {
                    _gpuEnabled = false;
                    return false;
                }

                CudaDevice.Initialize(deviceId);
                _gpuEnabled = true;
                return true;
            }
            catch
            {
                _gpuEnabled = false;
                return false;
            }
        }

        public static float[,] ParallelMatrixMultiplyTransposeA(float[,] a, float[,] b)
        {
            int p = a.GetLength(0);
            int m = a.GetLength(1);
            int pb = b.GetLength(0);
            int n = b.GetLength(1);

            if (p != pb)
            {
                throw new ArgumentException("Matrix dimensions do not match for transposed multiplication");
            }

            var result = new float[m, n];

            Parallel.For(0, m, GetParallelOptions(), row =>
            {
                for (int t = 0; t < p; t++)
                {
                    float aVal = a[t, row];
                    for (int col = 0; col < n; col++)
                    {
                        result[row, col] += aVal * b[t, col];
                    }
                }
            });

            return result;
        }

        public static float[,] ParallelSoftmaxRows(float[,] input)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            var output = new float[rows, cols];

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    if (input[i, j] > max) max = input[i, j];
                }

                float sum = 0;
                for (int j = 0; j < cols; j++)
                {
                    float e = MathF.Exp(input[i, j] - max);
                    output[i, j] = e;
                    sum += e;
                }

                for (int j = 0; j < cols; j++)
                {
                    output[i, j] /= sum;
                }
            });

            return output;
        }

        public static float[,] ParallelLayerNormRows(float[,] input, float[] gamma, float[] beta, float epsilon, out float[] mean, out float[] std)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            var output = new float[rows, cols];
            var localMean = new float[rows];
            var localStd = new float[rows];

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                float rowMean = 0;
                for (int j = 0; j < cols; j++)
                {
                    rowMean += input[i, j];
                }
                rowMean /= cols;
                localMean[i] = rowMean;

                float variance = 0;
                for (int j = 0; j < cols; j++)
                {
                    float diff = input[i, j] - rowMean;
                    variance += diff * diff;
                }
                variance /= cols;
                float rowStd = MathF.Sqrt(variance + epsilon);
                localStd[i] = rowStd;

                for (int j = 0; j < cols; j++)
                {
                    float normalized = (input[i, j] - rowMean) / rowStd;
                    output[i, j] = gamma[j] * normalized + beta[j];
                }
            });

            mean = localMean;
            std = localStd;
            return output;
        }

        public static void AdamUpdateAuto(float[,] parameters, float[,] gradients, float[,] m, float[,] v,
            float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2)
        {
            if (_gpuEnabled)
            {
                try
                {
                    CudaMath.AdamUpdate(parameters, gradients, m, v, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);
                    return;
                }
                catch
                {
                    _gpuEnabled = false;
                }
            }

            ParallelAdamUpdate(parameters, gradients, m, v, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);
        }

        public static void ParallelAdamUpdate(float[,] parameters, float[,] gradients, float[,] m, float[,] v,
            float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2)
        {
            int rows = parameters.GetLength(0);
            int cols = parameters.GetLength(1);

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    float g = gradients[i, j];

                    m[i, j] = beta1 * m[i, j] + (1 - beta1) * g;
                    v[i, j] = beta2 * v[i, j] + (1 - beta2) * g * g;

                    float mHat = m[i, j] / biasCorrection1;
                    float vHat = v[i, j] / biasCorrection2;

                    parameters[i, j] -= learningRate * mHat / (MathF.Sqrt(vHat) + epsilon);
                }
            });
        }

        public static void AdamUpdateAuto(float[] parameters, float[] gradients, float[] m, float[] v,
            float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2)
        {
            if (_gpuEnabled)
            {
                try
                {
                    CudaMath.AdamUpdate(parameters, gradients, m, v, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);
                    return;
                }
                catch
                {
                    _gpuEnabled = false;
                }
            }

            SequentialAdamUpdate(parameters, gradients, m, v, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);
        }

        public static void SequentialAdamUpdate(float[] parameters, float[] gradients, float[] m, float[] v,
            float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2)
        {
            for (int i = 0; i < parameters.Length; i++)
            {
                float g = gradients[i];

                m[i] = beta1 * m[i] + (1 - beta1) * g;
                v[i] = beta2 * v[i] + (1 - beta2) * g * g;

                float mHat = m[i] / biasCorrection1;
                float vHat = v[i] / biasCorrection2;

                parameters[i] -= learningRate * mHat / (MathF.Sqrt(vHat) + epsilon);
            }
        }

        public static float[,] ParallelMatrixMultiplyTransposeB(float[,] a, float[,] b, float scale = 1.0f)
        {
            int rowsA = a.GetLength(0);
            int colsA = a.GetLength(1);
            int rowsB = b.GetLength(0);
            int colsB = b.GetLength(1);

            if (colsA != colsB)
            {
                throw new ArgumentException("Matrix dimensions do not match for transposed multiplication");
            }

            var result = new float[rowsA, rowsB];

            Parallel.For(0, rowsA, GetParallelOptions(), i =>
            {
                for (int j = 0; j < rowsB; j++)
                {
                    float sum = 0.0f;
                    for (int p = 0; p < colsA; p++)
                    {
                        sum += a[i, p] * b[j, p];
                    }
                    result[i, j] = sum * scale;
                }
            });

            return result;
        }

        private static ParallelOptions GetParallelOptions()
        {
            return new ParallelOptions
            {
                MaxDegreeOfParallelism = _numThreads
            };
        }

        public static unsafe float[,] ParallelMatrixMultiply(float[,] a, float[,] b)
        {
            int rowsA = a.GetLength(0);
            int colsA = a.GetLength(1);
            int colsB = b.GetLength(1);

            if (colsA != b.GetLength(0))
            {
                throw new ArgumentException("Matrix dimensions do not match for multiplication");
            }

            var result = new float[rowsA, colsB];
            int vecSize = Vector<float>.Count;

            // Nota de rendimiento: se reordenan los bucles de i-j-k a i-k-j.
            // Con i-j-k, el acceso b[k,j] salta de columna en columna en cada
            // iteración de k (mala localidad de caché, ~1 cache miss por elemento).
            // Con i-k-j, para cada k se recorre la fila k de b de forma contigua
            // y se acumula sobre toda la fila resultado, lo cual además permite
            // vectorizar con SIMD (Vector<float>, 8 floats a la vez con AVX2).
            fixed (float* pa = a, pb = b, pr = result)
            {
                // Los lambdas de C# no pueden capturar variables de tipo puntero
                // directamente (CS1686/CS8909); se pasan como nint (entero) y se
                // vuelven a castear a puntero dentro del lambda.
                nint baseA = (nint)pa;
                nint baseB = (nint)pb;
                nint baseR = (nint)pr;

                Parallel.For(0, rowsA, GetParallelOptions(), i =>
                {
                    unsafe
                    {
                        float* rowA = (float*)baseA + (long)i * colsA;
                        float* rowR = (float*)baseR + (long)i * colsB;

                        for (int k = 0; k < colsA; k++)
                        {
                            float aVal = rowA[k];
                            var aVec = new Vector<float>(aVal);
                            float* rowB = (float*)baseB + (long)k * colsB;

                            int j = 0;
                            for (; j <= colsB - vecSize; j += vecSize)
                            {
                                var bVec = *(Vector<float>*)(rowB + j);
                                var rVec = *(Vector<float>*)(rowR + j);
                                rVec += aVec * bVec;
                                *(Vector<float>*)(rowR + j) = rVec;
                            }
                            for (; j < colsB; j++)
                            {
                                rowR[j] += aVal * rowB[j];
                            }
                        }
                    }
                });
            }

            return result;
        }

        public static float[,] ParallelMatrixAdd(float[,] a, float[,] b)
        {
            int rows = a.GetLength(0);
            int cols = a.GetLength(1);

            if (rows != b.GetLength(0) || cols != b.GetLength(1))
            {
                throw new ArgumentException("Matrix dimensions must match for addition");
            }

            var result = new float[rows, cols];

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = a[i, j] + b[i, j];
                }
            });

            return result;
        }

        public static float[,] ParallelMatrixScale(float[,] matrix, float scalar)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows, cols];

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = matrix[i, j] * scalar;
                }
            });

            return result;
        }

        public static void ParallelMatrixAddInPlace(float[,] target, float[,] source, float scale = 1.0f)
        {
            int rows = target.GetLength(0);
            int cols = target.GetLength(1);

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    target[i, j] += source[i, j] * scale;
                }
            });
        }

        public static float[] ParallelSoftmax(float[] input)
        {
            float max = input.Max();
            var exp = new float[input.Length];
            float sum = 0;

            Parallel.For(0, input.Length, GetParallelOptions(), i =>
            {
                exp[i] = (float)Math.Exp(input[i] - max);
            });

            for (int i = 0; i < exp.Length; i++)
            {
                sum += exp[i];
            }

            var result = new float[input.Length];
            Parallel.For(0, input.Length, GetParallelOptions(), i =>
            {
                result[i] = exp[i] / sum;
            });

            return result;
        }

        public static float[,] ParallelReLU(float[,] input)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            var result = new float[rows, cols];

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = Math.Max(0, input[i, j]);
                }
            });

            return result;
        }

        public static void ParallelClearMatrix(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = 0;
                }
            });
        }

        // ---------------------------------------------------------------
        public static unsafe float[,] SequentialMatrixMultiply(float[,] a, float[,] b)
        {
            int rowsA = a.GetLength(0);
            int colsA = a.GetLength(1);
            int colsB = b.GetLength(1);

            if (colsA != b.GetLength(0))
            {
                throw new ArgumentException("Matrix dimensions do not match for multiplication");
            }

            var result = new float[rowsA, colsB];
            int vecSize = Vector<float>.Count;

            fixed (float* pa = a, pb = b, pr = result)
            {
                for (int i = 0; i < rowsA; i++)
                {
                    float* rowA = pa + (long)i * colsA;
                    float* rowR = pr + (long)i * colsB;

                    for (int k = 0; k < colsA; k++)
                    {
                        float aVal = rowA[k];
                        var aVec = new Vector<float>(aVal);
                        float* rowB = pb + (long)k * colsB;

                        int j = 0;
                        for (; j <= colsB - vecSize; j += vecSize)
                        {
                            var bVec = *(Vector<float>*)(rowB + j);
                            var rVec = *(Vector<float>*)(rowR + j);
                            rVec += aVec * bVec;
                            *(Vector<float>*)(rowR + j) = rVec;
                        }
                        for (; j < colsB; j++)
                        {
                            rowR[j] += aVal * rowB[j];
                        }
                    }
                }
            }

            return result;
        }

        public static float[,] SequentialSoftmax2D(float[,] input)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);
            var output = new float[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                float max = float.NegativeInfinity;
                for (int j = 0; j < cols; j++)
                {
                    if (input[i, j] > max) max = input[i, j];
                }

                float sum = 0;
                for (int j = 0; j < cols; j++)
                {
                    output[i, j] = (float)Math.Exp(input[i, j] - max);
                    sum += output[i, j];
                }

                for (int j = 0; j < cols; j++)
                {
                    output[i, j] /= sum;
                }
            }

            return output;
        }

        public static float[,] ParallelTranspose(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[cols, rows];

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            });

            return result;
        }

        public static float ParallelSum(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var partialSums = new float[rows];

            Parallel.For(0, rows, GetParallelOptions(), i =>
            {
                float sum = 0;
                for (int j = 0; j < cols; j++)
                {
                    sum += matrix[i, j];
                }
                partialSums[i] = sum;
            });

            return partialSums.Sum();
        }

        // ---------------------------------------------------------------
        // Fase 4.1 - Operaciones por batch (batch como dimensión extra).
        // Representación: float[,,] con forma [batch, filas, columnas].
        // Cada operación produce el mismo resultado que aplicar la versión
        // no batcheada de arriba a cada elemento del batch por separado
        // (ver Matematicas_BatchOperationsTests para la prueba de equivalencia).
        // Se paraleliza sobre el índice combinado (batch * fila) en un único
        // Parallel.For, sin anidar, a propósito: revisar/optimizar el
        // paralelismo es tarea de la Fase 4.4, no de acá.
        // ---------------------------------------------------------------

        public static void SetBatchSlice(float[,,] batch, int batchIndex, float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            Buffer.BlockCopy(matrix, 0, batch, batchIndex * rows * cols * sizeof(float), rows * cols * sizeof(float));
        }

        public static float[,] FlattenBatch(float[,,] batch)
        {
            int batchSize = batch.GetLength(0);
            int seqLen = batch.GetLength(1);
            int dim = batch.GetLength(2);
            var flat = new float[batchSize * seqLen, dim];

            // batch[b, i, j] y flat[b*seqLen + i, j] comparten el mismo layout
            // row-major contiguo: aplanar es un único memcpy, no una copia
            // elemento a elemento (mismo caso que TransformerModel.FlattenBatch).
            Buffer.BlockCopy(batch, 0, flat, 0, batchSize * seqLen * dim * sizeof(float));

            return flat;
        }

        public static float[,,] UnflattenBatch(float[,] flat, int batchSize, int seqLen)
        {
            int dim = flat.GetLength(1);
            var batch = new float[batchSize, seqLen, dim];

            Buffer.BlockCopy(flat, 0, batch, 0, batchSize * seqLen * dim * sizeof(float));

            return batch;
        }

    }
}