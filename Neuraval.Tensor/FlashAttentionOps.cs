using System;
using System.Threading.Tasks;
using Neuraval.Tensor.Backends.Cpu;

namespace Neuraval.Tensor
{
    public static class FlashAttentionOps
    {
        public static Tensor Attend(
            Tensor queries, Tensor keys, Tensor values,
            bool causal, float scaleMultiplier, int queryOffset = 0)
        {
            if (queries.Rank != 2 || keys.Rank != 2 || values.Rank != 2)
            {
                throw new ArgumentException("FlashAttentionOps.Attend requiere tensores de rank 2");
            }

            int seqLenQ = queries.Shape[0];
            int headDim = queries.Shape[1];
            int seqLenK = keys.Shape[0];

            if (keys.Shape[1] != headDim || values.Shape[1] != headDim || values.Shape[0] != seqLenK)
            {
                throw new ArgumentException("Dimensiones incompatibles entre queries, keys y values");
            }

            if (queryOffset < 0)
            {
                throw new ArgumentException("queryOffset no puede ser negativo");
            }

            var output = new Tensor(new[] { seqLenQ, headDim }, queries.Device, queries.DType);

            var qBuf = queries.Buffer;
            var kBuf = keys.Buffer;
            var vBuf = values.Buffer;
            var outBuf = output.Buffer;

            Parallel.For(0, seqLenQ, new ParallelOptions { MaxDegreeOfParallelism = CpuBackendSelector.GetNumThreads() }, i =>
            {
                int qRow = i * headDim;
                int limit = causal ? Math.Min(queryOffset + i, seqLenK - 1) : seqLenK - 1;

                float runningMax = float.NegativeInfinity;
                float runningSum = 0f;
                var acc = new float[headDim];

                for (int j = 0; j <= limit; j++)
                {
                    int kRow = j * headDim;
                    float dot = 0f;

                    for (int d = 0; d < headDim; d++)
                    {
                        dot += qBuf[qRow + d] * kBuf[kRow + d];
                    }

                    dot *= scaleMultiplier;

                    float newMax = dot > runningMax ? dot : runningMax;
                    float correction = float.IsNegativeInfinity(runningMax) ? 0f : MathF.Exp(runningMax - newMax);
                    float weight = MathF.Exp(dot - newMax);

                    runningSum = runningSum * correction + weight;

                    int vRow = j * headDim;
                    for (int d = 0; d < headDim; d++)
                    {
                        acc[d] = acc[d] * correction + weight * vBuf[vRow + d];
                    }

                    runningMax = newMax;
                }

                float normalizer = runningSum > 0f ? 1.0f / runningSum : 0f;
                int outRow = i * headDim;

                for (int d = 0; d < headDim; d++)
                {
                    outBuf[outRow + d] = acc[d] * normalizer;
                }
            });

            return output;
        }
    }
}
