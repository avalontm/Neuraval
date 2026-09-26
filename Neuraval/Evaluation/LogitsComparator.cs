using System;

namespace Neuraval.Core.Evaluation
{
    public static class LogitsComparator
    {
        private const float RelativeErrorEpsilon = 1e-6f;

        public static LogitsComparisonReport Compare(float[] reference, float[] candidate)
        {
            if (reference == null)
                throw new ArgumentNullException(nameof(reference));

            if (candidate == null)
                throw new ArgumentNullException(nameof(candidate));

            if (reference.Length != candidate.Length)
                throw new ArgumentException("reference y candidate deben tener la misma longitud");

            if (reference.Length == 0)
                throw new ArgumentException("reference y candidate no pueden estar vacios");

            float maxAbsoluteError = 0f;
            double sumAbsoluteError = 0;
            double sumRelativeError = 0;
            double dotProduct = 0;
            double referenceNormSquared = 0;
            double candidateNormSquared = 0;

            for (int i = 0; i < reference.Length; i++)
            {
                float absoluteError = MathF.Abs(reference[i] - candidate[i]);

                if (absoluteError > maxAbsoluteError)
                    maxAbsoluteError = absoluteError;

                sumAbsoluteError += absoluteError;
                sumRelativeError += absoluteError / MathF.Max(MathF.Abs(reference[i]), RelativeErrorEpsilon);

                dotProduct += (double)reference[i] * candidate[i];
                referenceNormSquared += (double)reference[i] * reference[i];
                candidateNormSquared += (double)candidate[i] * candidate[i];
            }

            float meanAbsoluteError = (float)(sumAbsoluteError / reference.Length);
            float meanRelativeError = (float)(sumRelativeError / reference.Length);
            float cosineSimilarity = ComputeCosineSimilarity(dotProduct, referenceNormSquared, candidateNormSquared);

            return new LogitsComparisonReport(maxAbsoluteError, meanAbsoluteError, meanRelativeError, cosineSimilarity);
        }

        public static LogitsComparisonReport Compare(float[,] reference, float[,] candidate)
        {
            if (reference == null)
                throw new ArgumentNullException(nameof(reference));

            if (candidate == null)
                throw new ArgumentNullException(nameof(candidate));

            if (reference.GetLength(0) != candidate.GetLength(0) || reference.GetLength(1) != candidate.GetLength(1))
                throw new ArgumentException("reference y candidate deben tener el mismo shape");

            return Compare(Flatten(reference), Flatten(candidate));
        }

        private static float ComputeCosineSimilarity(double dotProduct, double referenceNormSquared, double candidateNormSquared)
        {
            bool referenceIsZero = referenceNormSquared == 0;
            bool candidateIsZero = candidateNormSquared == 0;

            if (referenceIsZero && candidateIsZero)
                return 1f;

            if (referenceIsZero || candidateIsZero)
                return 0f;

            return (float)(dotProduct / (Math.Sqrt(referenceNormSquared) * Math.Sqrt(candidateNormSquared)));
        }

        private static float[] Flatten(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            var result = new float[rows * cols];

            Buffer.BlockCopy(matrix, 0, result, 0, result.Length * sizeof(float));

            return result;
        }
    }
}
