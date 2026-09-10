using System;

namespace Neuraval.Tensor.Backends.Cpu
{
    public static class CpuBackendSelector
    {
        public static readonly CpuScalarBackend ScalarBackend = new CpuScalarBackend();
        public static readonly CpuSimdBackend SimdBackend = new CpuSimdBackend();
        public static readonly CpuParallelBackend ParallelBackend = new CpuParallelBackend();

        private static long _parallelWorkloadThreshold = 8_192;

        public static void ConfigureThreshold(long parallelWorkloadThreshold)
        {
            if (parallelWorkloadThreshold <= 0)
            {
                throw new ArgumentException("El umbral debe ser positivo");
            }

            _parallelWorkloadThreshold = parallelWorkloadThreshold;
        }

        public static void SetNumThreads(int numThreads)
        {
            ParallelBackend.SetNumThreads(numThreads);
        }

        public static int GetNumThreads()
        {
            return ParallelBackend.GetNumThreads();
        }

        public static ICpuTensorBackend SelectFor(long workload)
        {
            return workload <= _parallelWorkloadThreshold ? SimdBackend : ParallelBackend;
        }
    }
}
