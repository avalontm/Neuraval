using System;
using Neuraval.Core.Training.Precision;

namespace Neuraval.Core.Quantization
{
    public readonly struct PrecisionStageError
    {
        public float MaxAbsError { get; }
        public float MeanAbsError { get; }

        public PrecisionStageError(float maxAbsError, float meanAbsError)
        {
            MaxAbsError = maxAbsError;
            MeanAbsError = meanAbsError;
        }
    }

    public readonly struct PrecisionCascadeReport
    {
        public int Rows { get; }
        public int Cols { get; }
        public PrecisionStageError Fp16Error { get; }
        public PrecisionStageError Int8Error { get; }
        public long Fp32ByteSize { get; }
        public long Fp16ByteSize { get; }
        public long Int8ByteSize { get; }

        public PrecisionCascadeReport(
            int rows,
            int cols,
            PrecisionStageError fp16Error,
            PrecisionStageError int8Error,
            long fp32ByteSize,
            long fp16ByteSize,
            long int8ByteSize)
        {
            Rows = rows;
            Cols = cols;
            Fp16Error = fp16Error;
            Int8Error = int8Error;
            Fp32ByteSize = fp32ByteSize;
            Fp16ByteSize = fp16ByteSize;
            Int8ByteSize = int8ByteSize;
        }

        public static PrecisionCascadeReport Generate(float[] fp32Weights, int rows, int cols)
        {
            if (fp32Weights == null) throw new ArgumentNullException(nameof(fp32Weights));
            if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));
            if ((long)rows * cols != fp32Weights.Length)
            {
                throw new ArgumentException($"rows*cols ({(long)rows * cols}) no coincide con fp32Weights.Length ({fp32Weights.Length}).");
            }

            var fp16Weights = PrecisionConverter.ToFloat16(fp32Weights);
            var fp16Dequantized = PrecisionConverter.ToFloat32(fp16Weights);
            var fp16Error = Int8Quantizer.ComputeQuantizationError(fp32Weights, fp16Dequantized);

            var quantizedInt8 = Int8Quantizer.QuantizeRowSymmetric(fp32Weights, rows, cols);
            var int8Dequantized = Int8Quantizer.Dequantize(quantizedInt8);
            var int8Error = Int8Quantizer.ComputeQuantizationError(fp32Weights, int8Dequantized);

            long fp32ByteSize = (long)fp32Weights.Length * sizeof(float);
            long fp16ByteSize = (long)fp16Weights.Length * 2;

            return new PrecisionCascadeReport(
                rows,
                cols,
                new PrecisionStageError(fp16Error.MaxAbsError, fp16Error.MeanAbsError),
                new PrecisionStageError(int8Error.MaxAbsError, int8Error.MeanAbsError),
                fp32ByteSize,
                fp16ByteSize,
                quantizedInt8.QuantizedByteSize);
        }
    }
}
