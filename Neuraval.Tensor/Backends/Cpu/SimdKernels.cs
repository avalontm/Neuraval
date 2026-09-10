using System;
using System.Numerics;

namespace Neuraval.Tensor.Backends.Cpu
{
    internal static class SimdKernels
    {
        private static readonly int VectorSize = Vector<float>.Count;

        public static float DotProduct(float[] a, int aOffset, float[] b, int bOffset, int length)
        {
            var accumulator = Vector<float>.Zero;
            int i = 0;

            for (; i <= length - VectorSize; i += VectorSize)
            {
                var va = new Vector<float>(a, aOffset + i);
                var vb = new Vector<float>(b, bOffset + i);
                accumulator += va * vb;
            }

            float sum = Vector.Dot(accumulator, Vector<float>.One);

            for (; i < length; i++)
            {
                sum += a[aOffset + i] * b[bOffset + i];
            }

            return sum;
        }

        public static void Add(float[] a, float[] b, float[] result)
        {
            int length = result.Length;
            int i = 0;

            for (; i <= length - VectorSize; i += VectorSize)
            {
                var va = new Vector<float>(a, i);
                var vb = new Vector<float>(b, i);
                (va + vb).CopyTo(result, i);
            }

            for (; i < length; i++)
            {
                result[i] = a[i] + b[i];
            }
        }

        public static void AddInPlace(float[] target, float[] source, float scale)
        {
            int length = target.Length;
            var scaleVector = new Vector<float>(scale);
            int i = 0;

            for (; i <= length - VectorSize; i += VectorSize)
            {
                var vt = new Vector<float>(target, i);
                var vs = new Vector<float>(source, i);
                (vt + vs * scaleVector).CopyTo(target, i);
            }

            for (; i < length; i++)
            {
                target[i] += source[i] * scale;
            }
        }

        public static void Scale(float[] a, float scalar, float[] result)
        {
            int length = result.Length;
            var scalarVector = new Vector<float>(scalar);
            int i = 0;

            for (; i <= length - VectorSize; i += VectorSize)
            {
                var va = new Vector<float>(a, i);
                (va * scalarVector).CopyTo(result, i);
            }

            for (; i < length; i++)
            {
                result[i] = a[i] * scalar;
            }
        }

        public static void Multiply(float[] a, float[] b, float[] result)
        {
            int length = result.Length;
            int i = 0;

            for (; i <= length - VectorSize; i += VectorSize)
            {
                var va = new Vector<float>(a, i);
                var vb = new Vector<float>(b, i);
                (va * vb).CopyTo(result, i);
            }

            for (; i < length; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        public static void ReLU(float[] a, float[] result)
        {
            int length = result.Length;
            var zero = Vector<float>.Zero;
            int i = 0;

            for (; i <= length - VectorSize; i += VectorSize)
            {
                var va = new Vector<float>(a, i);
                Vector.Max(va, zero).CopyTo(result, i);
            }

            for (; i < length; i++)
            {
                result[i] = MathF.Max(0.0f, a[i]);
            }
        }

        public static void ReLUBackward(float[] grad, float[] activation, float[] result)
        {
            int length = result.Length;
            var zero = Vector<float>.Zero;
            int i = 0;

            for (; i <= length - VectorSize; i += VectorSize)
            {
                var va = new Vector<float>(activation, i);
                var vg = new Vector<float>(grad, i);
                var mask = Vector.GreaterThan(va, zero);
                Vector.ConditionalSelect(mask, vg, zero).CopyTo(result, i);
            }

            for (; i < length; i++)
            {
                result[i] = activation[i] > 0.0f ? grad[i] : 0.0f;
            }
        }

        public static float Sum(float[] a)
        {
            var accumulator = Vector<float>.Zero;
            int length = a.Length;
            int i = 0;

            for (; i <= length - VectorSize; i += VectorSize)
            {
                accumulator += new Vector<float>(a, i);
            }

            float sum = Vector.Dot(accumulator, Vector<float>.One);

            for (; i < length; i++)
            {
                sum += a[i];
            }

            return sum;
        }
    }
}
