using System;

namespace Neuraval.Core.Evaluation
{
    public sealed class ReferenceEquivalenceThresholds
    {
        public float MaxAbsoluteError { get; }

        public float MeanAbsoluteError { get; }

        public float MinCosineSimilarity { get; }

        private ReferenceEquivalenceThresholds(float maxAbsoluteError, float meanAbsoluteError, float minCosineSimilarity)
        {
            MaxAbsoluteError = maxAbsoluteError;
            MeanAbsoluteError = meanAbsoluteError;
            MinCosineSimilarity = minCosineSimilarity;
        }

        public static ReferenceEquivalenceThresholds Create(float maxAbsoluteError, float meanAbsoluteError, float minCosineSimilarity)
        {
            if (maxAbsoluteError <= 0f)
                throw new ArgumentOutOfRangeException(nameof(maxAbsoluteError));

            if (meanAbsoluteError <= 0f)
                throw new ArgumentOutOfRangeException(nameof(meanAbsoluteError));

            if (minCosineSimilarity <= 0f || minCosineSimilarity > 1f)
                throw new ArgumentOutOfRangeException(nameof(minCosineSimilarity));

            return new ReferenceEquivalenceThresholds(maxAbsoluteError, meanAbsoluteError, minCosineSimilarity);
        }

        public static ReferenceEquivalenceThresholds Default()
        {
            return new ReferenceEquivalenceThresholds(0.00008f, 0.000004f, 0.999999f);
        }
    }
}
