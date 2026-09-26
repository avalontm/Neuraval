using System;
using System.Collections.Generic;
using System.Globalization;

namespace Neuraval.Core.Evaluation
{
    public static class ReferenceEquivalenceChecker
    {
        public static EquivalenceVerdict Evaluate(LogitsComparisonReport report, ReferenceEquivalenceThresholds thresholds)
        {
            if (report == null)
                throw new ArgumentNullException(nameof(report));

            if (thresholds == null)
                throw new ArgumentNullException(nameof(thresholds));

            var reasons = new List<string>();

            if (report.MaxAbsoluteError > thresholds.MaxAbsoluteError)
            {
                reasons.Add(FormatReason("MaxAbsoluteError", report.MaxAbsoluteError, thresholds.MaxAbsoluteError));
            }

            if (report.MeanAbsoluteError > thresholds.MeanAbsoluteError)
            {
                reasons.Add(FormatReason("MeanAbsoluteError", report.MeanAbsoluteError, thresholds.MeanAbsoluteError));
            }

            if (report.CosineSimilarity < thresholds.MinCosineSimilarity)
            {
                reasons.Add(
                    $"CosineSimilarity {report.CosineSimilarity.ToString(CultureInfo.InvariantCulture)} " +
                    $"por debajo del minimo {thresholds.MinCosineSimilarity.ToString(CultureInfo.InvariantCulture)}");
            }

            return new EquivalenceVerdict(reasons.Count == 0, reasons);
        }

        private static string FormatReason(string metricName, float actual, float threshold)
        {
            return $"{metricName} {actual.ToString(CultureInfo.InvariantCulture)} " +
                   $"excede el umbral {threshold.ToString(CultureInfo.InvariantCulture)}";
        }
    }
}
