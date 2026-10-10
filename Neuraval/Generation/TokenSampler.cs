using System;
using System.Collections.Generic;

namespace Neuraval.Core.Generation
{
    public static class TokenSampler
    {
        private readonly record struct Candidate(int TokenId, float Logit);

        public static int SampleNext(float[] logits, GenerationOptions options, IReadOnlyList<int> generatedTokenIds, Random random)
        {
            if (logits == null)
                throw new ArgumentNullException(nameof(logits));

            if (options == null)
                throw new ArgumentNullException(nameof(options));

            if (generatedTokenIds == null)
                throw new ArgumentNullException(nameof(generatedTokenIds));

            if (random == null)
                throw new ArgumentNullException(nameof(random));

            if (options.Greedy)
            {
                return SampleGreedy(logits, generatedTokenIds, options.RepetitionPenalty);
            }

            return SampleFiltered(logits, options, generatedTokenIds, random);
        }

        private static int SampleGreedy(float[] logits, IReadOnlyList<int> generatedTokenIds, float repetitionPenalty)
        {
            HashSet<int>? repeated = null;
            if (repetitionPenalty != 1f && generatedTokenIds.Count > 0)
                repeated = new HashSet<int>(generatedTokenIds);

            int bestIndex = 0;
            float bestValue = AdjustLogit(logits[0], 0, repeated, repetitionPenalty);

            for (int i = 1; i < logits.Length; i++)
            {
                float value = AdjustLogit(logits[i], i, repeated, repetitionPenalty);
                if (value > bestValue)
                {
                    bestValue = value;
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        private static int SampleFiltered(
            float[] logits,
            GenerationOptions options,
            IReadOnlyList<int> generatedTokenIds,
            Random random)
        {
            int candidateLimit = Math.Min(options.TopK, logits.Length);
            HashSet<int>? repeated = null;
            if (options.RepetitionPenalty != 1f && generatedTokenIds.Count > 0)
                repeated = new HashSet<int>(generatedTokenIds);

            // Keep only the K best logits in O(vocabulary * log K), without cloning
            // and sorting the full vocabulary on every generated token.
            var heap = new PriorityQueue<Candidate, float>(candidateLimit);
            float temperatureScale = 1f / options.Temperature;
            for (int tokenId = 0; tokenId < logits.Length; tokenId++)
            {
                float value = AdjustLogit(logits[tokenId], tokenId, repeated, options.RepetitionPenalty) * temperatureScale;
                var candidate = new Candidate(tokenId, value);
                if (heap.Count < candidateLimit)
                {
                    heap.Enqueue(candidate, value);
                }
                else if (value > heap.Peek().Logit)
                {
                    heap.Dequeue();
                    heap.Enqueue(candidate, value);
                }
            }

            float threshold = heap.Peek().Logit;
            var candidates = new List<Candidate>(candidateLimit);
            // Preserve the conventional top-k tie behavior: all logits equal to the
            // cutoff remain eligible, even if that produces slightly more than K.
            for (int tokenId = 0; tokenId < logits.Length; tokenId++)
            {
                float value = AdjustLogit(logits[tokenId], tokenId, repeated, options.RepetitionPenalty) * temperatureScale;
                if (value >= threshold)
                    candidates.Add(new Candidate(tokenId, value));
            }

            candidates.Sort(static (left, right) => right.Logit.CompareTo(left.Logit));
            float maxLogit = candidates[0].Logit;
            float totalWeight = 0f;
            for (int i = 0; i < candidates.Count; i++)
            {
                float weight = MathF.Exp(candidates[i].Logit - maxLogit);
                candidates[i] = candidates[i] with { Logit = weight };
                totalWeight += weight;
            }

            int cutoff = candidates.Count;
            if (options.TopP < 1f)
            {
                float targetWeight = totalWeight * options.TopP;
                float cumulativeTopP = 0f;
                for (int i = 0; i < candidates.Count; i++)
                {
                    cumulativeTopP += candidates[i].Logit;
                    if (cumulativeTopP >= targetWeight)
                    {
                        cutoff = i + 1;
                        break;
                    }
                }
            }

            float retainedWeight = 0f;
            for (int i = 0; i < cutoff; i++)
                retainedWeight += candidates[i].Logit;

            float sample = (float)random.NextDouble() * retainedWeight;
            float cumulative = 0f;
            for (int i = 0; i < cutoff; i++)
            {
                cumulative += candidates[i].Logit;
                if (sample <= cumulative || i == cutoff - 1)
                    return candidates[i].TokenId;
            }

            return candidates[^1].TokenId;
        }

        private static float AdjustLogit(float value, int tokenId, HashSet<int>? repeated, float penalty)
        {
            if (repeated == null || !repeated.Contains(tokenId))
                return value;

            return value > 0f ? value / penalty : value * penalty;
        }
    }
}
