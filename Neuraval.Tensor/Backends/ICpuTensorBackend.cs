namespace Neuraval.Tensor.Backends
{
    public interface ICpuTensorBackend : ITensorBackend
    {
        Tensor Add(Tensor a, Tensor b);

        void AddInPlace(Tensor target, Tensor source, float scale);

        Tensor Scale(Tensor a, float scalar);

        Tensor Transpose(Tensor a, int rows, int cols);

        Tensor ReLU(Tensor a);

        Tensor ReLUBackward(Tensor grad, Tensor activation);

        Tensor Multiply(Tensor a, Tensor b);

        Tensor SumRows(Tensor a, int rows, int cols);

        float Sum(Tensor a);

        Tensor SoftmaxRowsBackward(Tensor gradOutput, Tensor softmaxOutput, int rows, int cols);
    }
}
