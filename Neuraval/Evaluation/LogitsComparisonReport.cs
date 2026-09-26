namespace Neuraval.Core.Evaluation
{
    public sealed class LogitsComparisonReport
    {
        public float MaxAbsoluteError { get; }

        public float MeanAbsoluteError { get; }

        public float MeanRelativeError { get; }

        public float CosineSimilarity { get; }

        public LogitsComparisonReport(float maxAbsoluteError, float meanAbsoluteError, float meanRelativeError, float cosineSimilarity)
        {
            MaxAbsoluteError = maxAbsoluteError;
            MeanAbsoluteError = meanAbsoluteError;
            MeanRelativeError = meanRelativeError;
            CosineSimilarity = cosineSimilarity;
        }
    }
}
