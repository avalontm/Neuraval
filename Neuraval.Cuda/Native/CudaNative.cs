using System.Runtime.InteropServices;

namespace Neuraval.Cuda.Native
{
    internal static class CudaNative
    {
        private const string LibraryName = "navcuda";

        [DllImport(LibraryName)]
        internal static extern int ncb_cuda_device_count();

        [DllImport(LibraryName)]
        internal static extern int ncb_cuda_get_device_name(int deviceId, byte[] buffer, int bufferLen);

        [DllImport(LibraryName)]
        internal static extern int ncb_cuda_init(int deviceId);

        [DllImport(LibraryName)]
        internal static extern IntPtr ncb_cuda_alloc(nuint bytes);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_free(IntPtr ptr);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_copy_h2d(IntPtr dst, float[] src, nuint bytes);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_copy_d2h(float[] dst, IntPtr src, nuint bytes);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_matmul_device(IntPtr devA, IntPtr devB, IntPtr devC, int m, int k, int n);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_matmul(float[] hostA, float[] hostB, float[] hostC, int m, int k, int n);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_matmul_transpose_b_device(IntPtr devA, IntPtr devB, IntPtr devC, int m, int k, int n, float scale);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_matmul_transpose_b(float[] hostA, float[] hostB, float[] hostC, int m, int k, int n, float scale);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_matmul_transpose_a_device(IntPtr devA, IntPtr devB, IntPtr devC, int p, int m, int n);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_matmul_transpose_a(float[] hostA, float[] hostB, float[] hostC, int p, int m, int n);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_cast_float_to_half_device(IntPtr devSrcFloat, IntPtr devDstHalf, int n);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_matmul_half_b_device(IntPtr devA, IntPtr devBHalf, IntPtr devC, int m, int k, int n);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_matmul_transpose_b_half_device(IntPtr devA, IntPtr devBHalf, IntPtr devC, int m, int k, int n, float scale);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_softmax_rows_device(IntPtr devInput, IntPtr devOutput, int rows, int cols);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_softmax_rows(float[] hostInput, float[] hostOutput, int rows, int cols);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_layernorm_rows_device(IntPtr devInput, IntPtr devGamma, IntPtr devBeta, IntPtr devOutput, IntPtr devMean, IntPtr devStd, int rows, int cols, float epsilon);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_layernorm_rows(float[] hostInput, float[] hostGamma, float[] hostBeta, float[] hostOutput, float[] hostMean, float[] hostStd, int rows, int cols, float epsilon);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_adam_update_device(IntPtr devParams, IntPtr devGrads, IntPtr devM, IntPtr devV, int n, float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_adam_update(float[] hostParams, float[] hostGrads, float[] hostM, float[] hostV, int n, float beta1, float beta2, float epsilon, float learningRate, float biasCorrection1, float biasCorrection2);

        [DllImport(LibraryName)]
        internal static extern void ncb_cuda_synchronize();

        [DllImport(LibraryName)]
        internal static extern int ncb_cuda_last_status();

        [DllImport(LibraryName)]
        internal static extern IntPtr ncb_cuda_status_message(int status);
    }
}
