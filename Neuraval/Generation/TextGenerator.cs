using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Neuraval.Core.Models;

namespace Neuraval.Core.Generation
{
    public sealed class TextGenerator
    {
        private readonly ModernDecoderModel _model;

        public TextGenerator(ModernDecoderModel model)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
        }

        public GenerationResult Generate(int[] promptTokenIds, GenerationOptions options, Random? random = null)
        {
            if (promptTokenIds == null || promptTokenIds.Length == 0)
                throw new ArgumentException("promptTokenIds no puede ser nulo ni vacío", nameof(promptTokenIds));

            if (options == null)
                throw new ArgumentNullException(nameof(options));

            var rng = random ?? (options.Seed.HasValue ? new Random(options.Seed.Value) : new Random());

            var sequence = new List<int>(promptTokenIds);
            var generatedTokenIds = new List<int>();
            var finishReason = GenerationFinishReason.MaxNewTokens;

            for (int step = 0; step < options.MaxNewTokens; step++)
            {
                var context = TrimToContextWindow(sequence);
                var logits = _model.Forward(context);
                var lastLogits = ExtractLastPositionLogits(logits);

                int nextToken = TokenSampler.SampleNext(lastLogits, options, generatedTokenIds, rng);

                generatedTokenIds.Add(nextToken);
                sequence.Add(nextToken);

                if (options.StopTokenIds.Contains(nextToken))
                {
                    finishReason = GenerationFinishReason.StopToken;
                    break;
                }
            }

            return GenerationResult.Create(generatedTokenIds, finishReason);
        }

        public CachedGenerationOutput GenerateWithCache(
            int[] promptTokenIds,
            GenerationOptions options,
            Random? random = null,
            Action<int>? onTokenGenerated = null,
            int? contextLimit = null)
        {
            if (promptTokenIds == null || promptTokenIds.Length == 0)
                throw new ArgumentException("promptTokenIds no puede ser nulo ni vacío", nameof(promptTokenIds));

            if (options == null)
                throw new ArgumentNullException(nameof(options));

            int maxLength = Math.Min(_model.MaxPositionEmbeddings, contextLimit ?? _model.MaxPositionEmbeddings);
            if (maxLength < 2)
                throw new ArgumentOutOfRangeException(nameof(contextLimit), "El contexto debe permitir al menos un token de entrada y uno de salida.");

            int generationCapacity = Math.Min(options.MaxNewTokens, maxLength - 1);
            int promptCapacity = maxLength - generationCapacity;
            var promptContext = TrimToContextWindow(new List<int>(promptTokenIds), promptCapacity);
            int capacity = promptContext.Length + generationCapacity;

            var cache = _model.CreateGenerationCache(capacity);
            var rng = random ?? (options.Seed.HasValue ? new Random(options.Seed.Value) : new Random());

            var generatedTokenIds = new List<int>();
            var finishReason = GenerationFinishReason.MaxNewTokens;

            var prefillStopwatch = Stopwatch.StartNew();
            var lastLogits = _model.ForwardIncrementalLastToken(promptContext, cache);
            prefillStopwatch.Stop();

            var decodeStopwatch = new Stopwatch();

            for (int step = 0; step < options.MaxNewTokens; step++)
            {
                if (cache.Length >= cache.Capacity)
                    break;

                decodeStopwatch.Start();
                int nextToken = TokenSampler.SampleNext(lastLogits, options, generatedTokenIds, rng);
                generatedTokenIds.Add(nextToken);

                if (options.StopTokenIds.Contains(nextToken))
                {
                    decodeStopwatch.Stop();
                    finishReason = GenerationFinishReason.StopToken;
                    break;
                }

                onTokenGenerated?.Invoke(nextToken);

                if (cache.Length >= cache.Capacity)
                {
                    decodeStopwatch.Stop();
                    break;
                }

                lastLogits = _model.ForwardIncrementalLastToken(new[] { nextToken }, cache);
                decodeStopwatch.Stop();
            }

            var result = GenerationResult.Create(generatedTokenIds, finishReason);
            var performance = GenerationPerformanceReport.Create(
                promptContext.Length,
                prefillStopwatch.Elapsed.TotalSeconds,
                generatedTokenIds.Count,
                decodeStopwatch.Elapsed.TotalSeconds,
                cache.EstimatedMemoryBytes);

            return CachedGenerationOutput.Create(result, performance);
        }

        private int[] TrimToContextWindow(List<int> sequence)
        {
            return TrimToContextWindow(sequence, _model.MaxPositionEmbeddings);
        }

        private static int[] TrimToContextWindow(List<int> sequence, int maxLength)
        {

            if (sequence.Count <= maxLength)
                return sequence.ToArray();

            return sequence.GetRange(sequence.Count - maxLength, maxLength).ToArray();
        }

        private static float[] ExtractLastPositionLogits(float[,] logits)
        {
            int lastPosition = logits.GetLength(0) - 1;
            int vocabSize = logits.GetLength(1);
            var result = new float[vocabSize];

            for (int i = 0; i < vocabSize; i++)
                result[i] = logits[lastPosition, i];

            return result;
        }
    }
}
