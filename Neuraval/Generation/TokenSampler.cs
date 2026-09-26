using System;
using System.Collections.Generic;
using Neuraval.Core.Utils;

namespace Neuraval.Core.Generation
{
    public static class TokenSampler
    {
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
                // BUGFIX: antes el modo greedy ignoraba por completo
                // options.RepetitionPenalty y hacía ArgMax(logits) sobre los logits
                // crudos. Greedy decoding sin penalización de repetición es la causa
                // clásica de que la generación caiga en un bucle que repite el mismo
                // fragmento una y otra vez (justo el síntoma reportado): en cuanto el
                // modelo entra en un ciclo de 1-2 tokens de alta probabilidad, nada le
                // impide repetirlo para siempre. Se clona el array para no mutar los
                // logits originales y se penaliza antes del argmax.
                var greedyLogits = (float[])logits.Clone();
                LogitsProcessor.ApplyRepetitionPenalty(greedyLogits, generatedTokenIds, options.RepetitionPenalty);
                return ArgMax(greedyLogits);
            }

            var working = (float[])logits.Clone();

            LogitsProcessor.ApplyRepetitionPenalty(working, generatedTokenIds, options.RepetitionPenalty);
            LogitsProcessor.ApplyTemperature(working, options.Temperature);
            LogitsProcessor.FilterTopK(working, options.TopK);

            var probabilities = Matematicas.ParallelSoftmax(working);

            LogitsProcessor.FilterTopP(probabilities, options.TopP);

            return SampleFromDistribution(probabilities, random);
        }

        private static int ArgMax(float[] logits)
        {
            int bestIndex = 0;
            float bestValue = logits[0];

            for (int i = 1; i < logits.Length; i++)
            {
                if (logits[i] > bestValue)
                {
                    bestValue = logits[i];
                    bestIndex = i;
                }
            }

            return bestIndex;
        }

        private static int SampleFromDistribution(float[] probabilities, Random random)
        {
            float total = 0f;
            for (int i = 0; i < probabilities.Length; i++)
                total += probabilities[i];

            if (total <= 0f)
                return ArgMax(probabilities);

            float sample = (float)(random.NextDouble() * total);
            float cumulative = 0f;

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (sample <= cumulative)
                    return i;
            }

            return probabilities.Length - 1;
        }
    }
}
