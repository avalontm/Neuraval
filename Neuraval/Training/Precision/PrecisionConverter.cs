using System;

namespace Neuraval.Core.Training.Precision
{
    public static class PrecisionConverter
    {
        public static Half[] ToFloat16(float[] source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            var result = new Half[source.Length];

            for (int i = 0; i < source.Length; i++)
            {
                result[i] = (Half)source[i];
            }

            return result;
        }

        public static float[] ToFloat32(Half[] source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            var result = new float[source.Length];

            for (int i = 0; i < source.Length; i++)
            {
                result[i] = (float)source[i];
            }

            return result;
        }

        public static BFloat16[] ToBFloat16(float[] source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            var result = new BFloat16[source.Length];

            for (int i = 0; i < source.Length; i++)
            {
                result[i] = BFloat16.FromSingle(source[i]);
            }

            return result;
        }

        public static float[] ToFloat32(BFloat16[] source)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            var result = new float[source.Length];

            for (int i = 0; i < source.Length; i++)
            {
                result[i] = source[i].ToSingle();
            }

            return result;
        }

        public static float[] RoundTrip(float[] source, ComputeDType dtype)
        {
            if (source == null)
                throw new ArgumentNullException(nameof(source));

            switch (dtype)
            {
                case ComputeDType.Fp32:
                    return (float[])source.Clone();
                case ComputeDType.Fp16:
                    return ToFloat32(ToFloat16(source));
                case ComputeDType.Bf16:
                    return ToFloat32(ToBFloat16(source));
                default:
                    throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "dtype no reconocido");
            }
        }

        public static float MaxAbsoluteError(float[] original, float[] roundTripped)
        {
            if (original == null)
                throw new ArgumentNullException(nameof(original));

            if (roundTripped == null)
                throw new ArgumentNullException(nameof(roundTripped));

            if (original.Length != roundTripped.Length)
                throw new ArgumentException("Las longitudes de los arreglos no coinciden");

            float maxError = 0f;

            for (int i = 0; i < original.Length; i++)
            {
                float error = MathF.Abs(original[i] - roundTripped[i]);

                if (error > maxError)
                    maxError = error;
            }

            return maxError;
        }
    }
}
