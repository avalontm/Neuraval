namespace Neuraval.Cuda
{
    public class CudaException : Exception
    {
        public int StatusCode { get; }

        public CudaException(int statusCode, string message) : base(message)
        {
            StatusCode = statusCode;
        }
    }
}
