using System;
using System.Threading.Tasks;
using Neuraval.Core.Utils;

namespace Neuraval.Core.Models.RoPE
{
    public sealed class RotaryEmbedding
    {
        private readonly RotaryEmbeddingCache _cache;

        public int HeadDim => _cache.HeadDim;

        public RotaryEmbedding(RotaryEmbeddingCache cache)
        {
            _cache = cache ?? throw new ArgumentNullException(nameof(cache));
        }

        public RotaryEmbedding(RotaryConfig config) : this(new RotaryEmbeddingCache(config))
        {
        }

        public float[,] Apply(float[,] x, int positionOffset = 0)
        {
            ValidateHeadDim(x.GetLength(1));

            int seqLen = x.GetLength(0);
            int headDim = x.GetLength(1);
            var output = new float[seqLen, headDim];

            Parallel.For(0, seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, s =>
            {
                RotateRow(x, output, s, positionOffset + s, forward: true);
            });

            return output;
        }

        public float[,] ApplyBackward(float[,] gradOutput, int positionOffset = 0)
        {
            ValidateHeadDim(gradOutput.GetLength(1));

            int seqLen = gradOutput.GetLength(0);
            int headDim = gradOutput.GetLength(1);
            var gradInput = new float[seqLen, headDim];

            Parallel.For(0, seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, s =>
            {
                RotateRow(gradOutput, gradInput, s, positionOffset + s, forward: false);
            });

            return gradInput;
        }

        public float[,,] Apply(float[,,] x, int positionOffset = 0)
        {
            ValidateHeadDim(x.GetLength(2));

            int batchSize = x.GetLength(0);
            int seqLen = x.GetLength(1);
            int headDim = x.GetLength(2);
            var output = new float[batchSize, seqLen, headDim];

            Parallel.For(0, batchSize * seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, flat =>
            {
                int b = flat / seqLen;
                int s = flat % seqLen;
                RotateRow3D(x, output, b, s, positionOffset + s, forward: true);
            });

            return output;
        }

        public float[,,] ApplyBackward(float[,,] gradOutput, int positionOffset = 0)
        {
            ValidateHeadDim(gradOutput.GetLength(2));

            int batchSize = gradOutput.GetLength(0);
            int seqLen = gradOutput.GetLength(1);
            int headDim = gradOutput.GetLength(2);
            var gradInput = new float[batchSize, seqLen, headDim];

            Parallel.For(0, batchSize * seqLen, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, flat =>
            {
                int b = flat / seqLen;
                int s = flat % seqLen;
                RotateRow3D(gradOutput, gradInput, b, s, positionOffset + s, forward: false);
            });

            return gradInput;
        }

        public float[,,,] Apply(float[,,,] x, int positionOffset = 0)
        {
            ValidateHeadDim(x.GetLength(3));

            int batchSize = x.GetLength(0);
            int seqLen = x.GetLength(1);
            int numHeads = x.GetLength(2);
            int headDim = x.GetLength(3);
            var output = new float[batchSize, seqLen, numHeads, headDim];

            Parallel.For(0, batchSize * seqLen * numHeads, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, flat =>
            {
                int h = flat % numHeads;
                int rest = flat / numHeads;
                int s = rest % seqLen;
                int b = rest / seqLen;
                RotateRow4D(x, output, b, s, h, positionOffset + s, forward: true);
            });

            return output;
        }

        public float[,,,] ApplyBackward(float[,,,] gradOutput, int positionOffset = 0)
        {
            ValidateHeadDim(gradOutput.GetLength(3));

            int batchSize = gradOutput.GetLength(0);
            int seqLen = gradOutput.GetLength(1);
            int numHeads = gradOutput.GetLength(2);
            int headDim = gradOutput.GetLength(3);
            var gradInput = new float[batchSize, seqLen, numHeads, headDim];

            Parallel.For(0, batchSize * seqLen * numHeads, new ParallelOptions { MaxDegreeOfParallelism = Matematicas.GetNumThreads() }, flat =>
            {
                int h = flat % numHeads;
                int rest = flat / numHeads;
                int s = rest % seqLen;
                int b = rest / seqLen;
                RotateRow4D(gradOutput, gradInput, b, s, h, positionOffset + s, forward: false);
            });

            return gradInput;
        }

        private void RotateRow(float[,] input, float[,] output, int row, int position, bool forward)
        {
            int halfDim = HeadDim / 2;

            for (int i = 0; i < halfDim; i++)
            {
                float cos = _cache.CosAt(position, i);
                float sin = _cache.SinAt(position, i);

                float x1 = input[row, i];
                float x2 = input[row, i + halfDim];

                if (forward)
                {
                    output[row, i] = x1 * cos - x2 * sin;
                    output[row, i + halfDim] = x2 * cos + x1 * sin;
                }
                else
                {
                    output[row, i] = x1 * cos + x2 * sin;
                    output[row, i + halfDim] = x2 * cos - x1 * sin;
                }
            }
        }

        private void RotateRow3D(float[,,] input, float[,,] output, int batch, int seq, int position, bool forward)
        {
            int halfDim = HeadDim / 2;

            for (int i = 0; i < halfDim; i++)
            {
                float cos = _cache.CosAt(position, i);
                float sin = _cache.SinAt(position, i);

                float x1 = input[batch, seq, i];
                float x2 = input[batch, seq, i + halfDim];

                if (forward)
                {
                    output[batch, seq, i] = x1 * cos - x2 * sin;
                    output[batch, seq, i + halfDim] = x2 * cos + x1 * sin;
                }
                else
                {
                    output[batch, seq, i] = x1 * cos + x2 * sin;
                    output[batch, seq, i + halfDim] = x2 * cos - x1 * sin;
                }
            }
        }

        private void RotateRow4D(float[,,,] input, float[,,,] output, int batch, int seq, int head, int position, bool forward)
        {
            int halfDim = HeadDim / 2;

            for (int i = 0; i < halfDim; i++)
            {
                float cos = _cache.CosAt(position, i);
                float sin = _cache.SinAt(position, i);

                float x1 = input[batch, seq, head, i];
                float x2 = input[batch, seq, head, i + halfDim];

                if (forward)
                {
                    output[batch, seq, head, i] = x1 * cos - x2 * sin;
                    output[batch, seq, head, i + halfDim] = x2 * cos + x1 * sin;
                }
                else
                {
                    output[batch, seq, head, i] = x1 * cos + x2 * sin;
                    output[batch, seq, head, i + halfDim] = x2 * cos - x1 * sin;
                }
            }
        }

        private void ValidateHeadDim(int dim)
        {
            if (dim != HeadDim)
                throw new ArgumentException($"La última dimensión ({dim}) no coincide con HeadDim ({HeadDim})");
        }
    }
}
