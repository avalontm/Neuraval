using Neuraval.Cuda;

namespace Neuraval.Tensor.Backends
{
    public interface ITensorBackend
    {
        Tensor MatMul(Tensor a, Tensor b, int m, int k, int n);

        Tensor MatMulTransposeB(Tensor a, Tensor b, int m, int k, int n, float scale);

        Tensor MatMulTransposeA(Tensor a, Tensor b, int k, int m, int n);

        Tensor MatMulCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n);

        Tensor MatMulTransposeBCachedB(Tensor a, float[,] weights, CudaWeightCache weightCache, int m, int k, int n, float scale);

        Tensor SoftmaxRows(Tensor a, int rows, int cols);

        Tensor LayerNormRows(Tensor a, Tensor gamma, Tensor beta, int rows, int cols, float epsilon, out float[] mean, out float[] std);
    }
}
