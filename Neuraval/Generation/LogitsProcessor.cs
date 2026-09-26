using System;
using System.Collections.Generic;

namespace Neuraval.Core.Generation
{
    public static class LogitsProcessor
    {
        public static void ApplyRepetitionPenalty(float[] logits, IReadOnlyCollection<int> generatedTokenIds, float penalty)
        {
            if (logits == null)
                throw new ArgumentNullException(nameof(logits));

            if (generatedTokenIds == null)
                throw new ArgumentNullException(nameof(generatedTokenIds));

            if (penalty <= 0f)
                throw new ArgumentOutOfRangeException(nameof(penalty));

            if (penalty == 1f || generatedTokenIds.Count == 0)
                return;

            var penalized = new HashSet<int>();

            foreach (var tokenId in generatedTokenIds)
            {
                if (tokenId < 0 || tokenId >= logits.Length)
                    continue;

                if (!penalized.Add(tokenId))
                    continue;

                logits[tokenId] = logits[tokenId] > 0f
                    ? logits[tokenId] / penalty
                    : logits[tokenId] * penalty;
            }
        }

        public static void ApplyTemperature(float[] logits, float temperature)
        {
            if (logits == null)
                throw new ArgumentNullException(nameof(logits));

            if (temperature <= 0f)
                throw new ArgumentOutOfRangeException(nameof(temperature));

            if (temperature == 1f)
                return;

            for (int i = 0; i < logits.Length; i++)
                logits[i] /= temperature;
        }

        public static void FilterTopK(float[] logits, int topK)
        {
            if (logits == null)
                throw new ArgumentNullException(nameof(logits));

            if (topK <= 0)
                throw new ArgumentOutOfRangeException(nameof(topK));

            if (topK >= logits.Length)
                return;

            var sorted = (float[])logits.Clone();
            Array.Sort(sorted);
            float threshold = sorted[sorted.Length - topK];

            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] < threshold)
                    logits[i] = float.NegativeInfinity;
            }
        }

        public static void FilterTopP(float[] probabilities, float topP)
        {
            if (probabilities == null)
                throw new ArgumentNullException(nameof(probabilities));

            if (topP <= 0f || topP > 1f)
                throw new ArgumentOutOfRangeException(nameof(topP));

            if (topP >= 1f)
                return;

            var order = new int[probabilities.Length];
            for (int i = 0; i < order.Length; i++)
                order[i] = i;

            Array.Sort(order, (a, b) => probabilities[b].CompareTo(probabilities[a]));

            int cutoff = order.Length;
            float cumulative = 0f;

            for (int i = 0; i < order.Length; i++)
            {
                cumulative += probabilities[order[i]];
                if (cumulative >= topP)
                {
                    cutoff = i + 1;
                    break;
                }
            }

            float retainedSum = 0f;
            for (int i = 0; i < cutoff; i++)
                retainedSum += probabilities[order[i]];

            var filtered = new float[probabilities.Length];
            for (int i = 0; i < cutoff; i++)
                filtered[order[i]] = retainedSum > 0f ? probabilities[order[i]] / retainedSum : 0f;

            Array.Copy(filtered, probabilities, probabilities.Length);
        }
    }
}
