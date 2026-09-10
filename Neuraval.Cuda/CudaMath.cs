using Neuraval.Cuda.Native;

namespace Neuraval.Cuda
{
    public static class CudaMath
    {
        public static float[,] MatrixMultiply(float[,] a, float[,] b)
        {
            int m = a.GetLength(0);
            int k = a.GetLength(1);
            int kb = b.GetLength(0);
            int n = b.GetLength(1);

            if (k != kb)
            {
                throw new ArgumentException("Matrix dimensions do not match for multiplication");
            }

            var flatA = new float[m * k];
            var flatB = new float[k * n];
            var flatC = new float[m * n];

            Buffer.BlockCopy(a, 0, flatA, 0, flatA.Length * sizeof(float));
            Buffer.BlockCopy(b, 0, flatB, 0, flatB.Length * sizeof(float));

            CudaNative.ncb_cuda_matmul(flatA, flatB, flatC, m, k, n);
            CudaDevice.ThrowIfError();

            var result = new float[m, n];
            Buffer.BlockCopy(flatC, 0, result, 0, flatC.Length * sizeof(float));

            return result;
        }

        public static CudaBuffer MatrixMultiply(CudaBuffer a, CudaBuffer b, int m, int k, int n)
        {
            var result = CudaBuffer.Allocate(m * n);
            CudaNative.ncb_cuda_matmul_device(a.DevicePointer, b.DevicePointer, result.DevicePointer, m, k, n);
            CudaDevice.ThrowIfError();
            return result;
        }

        public static float[] MatrixMultiply(float[] a, float[] b, int m, int k, int n)
        {
            var c = new float[m * n];
            CudaNative.ncb_cuda_matmul(a, b, c, m, k, n);
            CudaDevice.ThrowIfError();
            return c;
        }

        public static float[,] MatrixMultiplyTransposeB(float[,] a, float[,] b, float scale = 1.0f)
        {
            int m = a.GetLength(0);
            int k = a.GetLength(1);
            int n = b.GetLength(0);
            int kb = b.GetLength(1);

            if (k != kb)
            {
                throw new ArgumentException("Matrix dimensions do not match for transposed multiplication");
            }

            var flatA = new float[m * k];
            var flatB = new float[n * k];
            var flatC = new float[m * n];

            Buffer.BlockCopy(a, 0, flatA, 0, flatA.Length * sizeof(float));
            Buffer.BlockCopy(b, 0, flatB, 0, flatB.Length * sizeof(float));

            CudaNative.ncb_cuda_matmul_transpose_b(flatA, flatB, flatC, m, k, n, scale);
            CudaDevice.ThrowIfError();

            var result = new float[m, n];
            Buffer.BlockCopy(flatC, 0, result, 0, flatC.Length * sizeof(float));

            return result;
        }

        public static CudaBuffer MatrixMultiplyTransposeB(CudaBuffer a, CudaBuffer b, int m, int k, int n, float scale = 1.0f)
        {
            var result = CudaBuffer.Allocate(m * n);
            CudaNative.ncb_cuda_matmul_transpose_b_device(a.DevicePointer, b.DevicePointer, result.DevicePointer, m, k, n, scale);
            CudaDevice.ThrowIfError();
            return result;
        }

        public static float[] MatrixMultiplyTransposeB(float[] a, float[] b, int m, int k, int n, float scale = 1.0f)
        {
            var c = new float[m * n];
            CudaNative.ncb_cuda_matmul_transpose_b(a, b, c, m, k, n, scale);
            CudaDevice.ThrowIfError();
            return c;
        }

        public static float[,] MatrixMultiplyTransposeA(float[,] a, float[,] b)
        {
            int p = a.GetLength(0);
            int m = a.GetLength(1);
            int pb = b.GetLength(0);
            int n = b.GetLength(1);

            if (p != pb)
            {
                throw new ArgumentException("Matrix dimensions do not match for transposed multiplication");
            }

            var flatA = new float[p * m];
            var flatB = new float[p * n];
            var flatC = new float[m * n];

            Buffer.BlockCopy(a, 0, flatA, 0, flatA.Length * sizeof(float));
            Buffer.BlockCopy(b, 0, flatB, 0, flatB.Length * sizeof(float));

            CudaNative.ncb_cuda_matmul_transpose_a(flatA, flatB, flatC, p, m, n);
            CudaDevice.ThrowIfError();

            var result = new float[m, n];
            Buffer.BlockCopy(flatC, 0, result, 0, flatC.Length * sizeof(float));

            return result;
        }

        public static CudaBuffer MatrixMultiplyTransposeA(CudaBuffer a, CudaBuffer b, int p, int m, int n)
        {
            var result = CudaBuffer.Allocate(m * n);
            CudaNative.ncb_cuda_matmul_transpose_a_device(a.DevicePointer, b.DevicePointer, result.DevicePointer, p, m, n);
            CudaDevice.ThrowIfError();
            return result;
        }

        public static float[] MatrixMultiplyTransposeA(float[] a, float[] b, int p, int m, int n)
        {
            var c = new float[m * n];
            CudaNative.ncb_cuda_matmul_transpose_a(a, b, c, p, m, n);
            CudaDevice.ThrowIfError();
            return c;
        }

        public static float[,] SoftmaxRows(float[,] input)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);

            var flatIn = new float[rows * cols];
            var flatOut = new float[rows * cols];

            Buffer.BlockCopy(input, 0, flatIn, 0, flatIn.Length * sizeof(float));

            CudaNative.ncb_cuda_softmax_rows(flatIn, flatOut, rows, cols);
            CudaDevice.ThrowIfError();

            var result = new float[rows, cols];
            Buffer.BlockCopy(flatOut, 0, result, 0, flatOut.Length * sizeof(float));

            return result;
        }

        public static float[] SoftmaxRows(float[] input, int rows, int cols)
        {
            var output = new float[rows * cols];
            CudaNative.ncb_cuda_softmax_rows(input, output, rows, cols);
            CudaDevice.ThrowIfError();
            return output;
        }

        public static float[] LayerNormRows(float[] input, float[] gamma, float[] beta, int rows, int cols, float epsilon, out float[] mean, out float[] std)
        {
            if (gamma.Length != cols || beta.Length != cols)
            {
                throw new ArgumentException("Gamma/beta length must match the normalized dimension");
            }

            var output = new float[rows * cols];
            mean = new float[rows];
            std = new float[rows];

            CudaNative.ncb_cuda_layernorm_rows(input, gamma, beta, output, mean, std, rows, cols, epsilon);
            CudaDevice.ThrowIfError();

            return output;
        }

        public static float[,] LayerNormRows(float[,] input, float[] gamma, float[] beta, float epsilon, out float[] mean, out float[] std)
        {
            int rows = input.GetLength(0);
            int cols = input.GetLength(1);

            if (gamma.Length != cols || beta.Length != cols)
            {
                throw new ArgumentException("Gamma/beta length must match the normalized dimension");
            }

            var flatIn = new float[rows * cols];
            var flatOut = new float[rows * cols];
            mean = new float[rows];
            std = new float[rows];

            Buffer.BlockCopy(input, 0, flatIn, 0, flatIn.Length * sizeof(float));

            CudaNative.ncb_cuda_layernorm_rows(flatIn, gamma, beta, flatOut, mean, std, rows, cols, epsilon);
            CudaDevice.ThrowIfError();

            var result = new float[rows, cols];
            Buffer.BlockCopy(flatOut, 0, result, 0, flatOut.Length * sizeof(float));

            return result;
        }

        public static void AdamUpdate(float[,] parameters, float[,] gradients, float[,] m, float[,] v,
            float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2)
        {
            int n = parameters.Length;

            var flatParams = new float[n];
            var flatGrads = new float[n];
            var flatM = new float[n];
            var flatV = new float[n];

            Buffer.BlockCopy(parameters, 0, flatParams, 0, n * sizeof(float));
            Buffer.BlockCopy(gradients, 0, flatGrads, 0, n * sizeof(float));
            Buffer.BlockCopy(m, 0, flatM, 0, n * sizeof(float));
            Buffer.BlockCopy(v, 0, flatV, 0, n * sizeof(float));

            CudaNative.ncb_cuda_adam_update(flatParams, flatGrads, flatM, flatV, n, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);
            CudaDevice.ThrowIfError();

            Buffer.BlockCopy(flatParams, 0, parameters, 0, n * sizeof(float));
            Buffer.BlockCopy(flatM, 0, m, 0, n * sizeof(float));
            Buffer.BlockCopy(flatV, 0, v, 0, n * sizeof(float));
        }

        public static void AdamUpdate(float[] parameters, float[] gradients, float[] m, float[] v,
            float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2)
        {
            int n = parameters.Length;

            CudaNative.ncb_cuda_adam_update(parameters, gradients, m, v, n, beta1, beta2, epsilon, learningRate, biasCorrection1, biasCorrection2);
            CudaDevice.ThrowIfError();
        }

        public static float[,] MatrixMultiplyCachedB(float[,] a, float[,] weights, CudaWeightCache weightCache)
        {
            int m = a.GetLength(0);
            int k = a.GetLength(1);
            int n = weights.GetLength(1);

            if (weights.GetLength(0) != k)
            {
                throw new ArgumentException("Matrix dimensions do not match for multiplication");
            }

            var bBuffer = weightCache.GetOrUpload(weights);

            var flatA = new float[m * k];
            Buffer.BlockCopy(a, 0, flatA, 0, flatA.Length * sizeof(float));

            using var aBuffer = CudaBuffer.Upload(flatA);
            using var cBuffer = CudaBuffer.Allocate(m * n);

            CudaNative.ncb_cuda_matmul_device(aBuffer.DevicePointer, bBuffer.DevicePointer, cBuffer.DevicePointer, m, k, n);
            CudaDevice.ThrowIfError();

            var flatC = cBuffer.Download();
            var result = new float[m, n];
            Buffer.BlockCopy(flatC, 0, result, 0, flatC.Length * sizeof(float));

            return result;
        }

        public static float[,] MatrixMultiplyTransposeBCachedB(float[,] a, float[,] weights, CudaWeightCache weightCache, float scale = 1.0f)
        {
            int m = a.GetLength(0);
            int k = a.GetLength(1);
            int n = weights.GetLength(0);

            if (weights.GetLength(1) != k)
            {
                throw new ArgumentException("Matrix dimensions do not match for transposed multiplication");
            }

            var bBuffer = weightCache.GetOrUpload(weights);

            var flatA = new float[m * k];
            Buffer.BlockCopy(a, 0, flatA, 0, flatA.Length * sizeof(float));

            using var aBuffer = CudaBuffer.Upload(flatA);
            using var cBuffer = CudaBuffer.Allocate(m * n);

            CudaNative.ncb_cuda_matmul_transpose_b_device(aBuffer.DevicePointer, bBuffer.DevicePointer, cBuffer.DevicePointer, m, k, n, scale);
            CudaDevice.ThrowIfError();

            var flatC = cBuffer.Download();
            var result = new float[m, n];
            Buffer.BlockCopy(flatC, 0, result, 0, flatC.Length * sizeof(float));

            return result;
        }

        public static float[] MatrixMultiplyCachedB(float[] a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n)
        {
            if (weights.GetLength(0) != k || weights.GetLength(1) != n)
            {
                throw new ArgumentException("Matrix dimensions do not match for multiplication");
            }

            var bBuffer = weightCache.GetOrUpload(weights);

            using var aBuffer = CudaBuffer.Upload(a);
            using var cBuffer = CudaBuffer.Allocate(m * n);

            CudaNative.ncb_cuda_matmul_device(aBuffer.DevicePointer, bBuffer.DevicePointer, cBuffer.DevicePointer, m, k, n);
            CudaDevice.ThrowIfError();

            return cBuffer.Download();
        }

        public static float[] MatrixMultiplyTransposeBCachedB(float[] a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n, float scale = 1.0f)
        {
            if (weights.GetLength(0) != n || weights.GetLength(1) != k)
            {
                throw new ArgumentException("Matrix dimensions do not match for transposed multiplication");
            }

            var bBuffer = weightCache.GetOrUpload(weights);

            using var aBuffer = CudaBuffer.Upload(a);
            using var cBuffer = CudaBuffer.Allocate(m * n);

            CudaNative.ncb_cuda_matmul_transpose_b_device(aBuffer.DevicePointer, bBuffer.DevicePointer, cBuffer.DevicePointer, m, k, n, scale);
            CudaDevice.ThrowIfError();

            return cBuffer.Download();
        }

        public static float[] MatrixMultiplyCachedBHalf(float[] a, float[,] weights, CudaWeightCacheFp16 weightCache, int m, int k, int n)
        {
            if (weights.GetLength(0) != k || weights.GetLength(1) != n)
            {
                throw new ArgumentException("Matrix dimensions do not match for multiplication");
            }

            var bBuffer = weightCache.GetOrUpload(weights);

            using var aBuffer = CudaBuffer.Upload(a);
            using var cBuffer = CudaBuffer.Allocate(m * n);

            CudaNative.ncb_cuda_matmul_half_b_device(aBuffer.DevicePointer, bBuffer.DevicePointer, cBuffer.DevicePointer, m, k, n);
            CudaDevice.ThrowIfError();

            return cBuffer.Download();
        }

        public static float[] MatrixMultiplyTransposeBCachedBHalf(float[] a, float[,] weights, CudaWeightCacheFp16 weightCache, int m, int k, int n, float scale = 1.0f)
        {
            if (weights.GetLength(0) != n || weights.GetLength(1) != k)
            {
                throw new ArgumentException("Matrix dimensions do not match for transposed multiplication");
            }

            var bBuffer = weightCache.GetOrUpload(weights);

            using var aBuffer = CudaBuffer.Upload(a);
            using var cBuffer = CudaBuffer.Allocate(m * n);

            CudaNative.ncb_cuda_matmul_transpose_b_half_device(aBuffer.DevicePointer, bBuffer.DevicePointer, cBuffer.DevicePointer, m, k, n, scale);
            CudaDevice.ThrowIfError();

            return cBuffer.Download();
        }
    }
}
