using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Neuraval.Core.Services
{
    public class DatasetLoader
    {
        private readonly ITokenizer _tokenizer;
        private readonly int _maxSequenceLength;
        private readonly Random _shuffleRandom;

        public DatasetLoader(ITokenizer tokenizer, int maxSequenceLength = 128, int shuffleSeed = 42)
        {
            _tokenizer = tokenizer;
            _maxSequenceLength = maxSequenceLength;
            _shuffleRandom = new Random(shuffleSeed);
        }

        public (List<int[]> inputs, List<int[]> targets) LoadConversationData(
            List<ConversationPair> conversations,
            bool shuffle = true)
        {
            var inputs = new List<int[]>();
            var targets = new List<int[]>();

            foreach (var conv in conversations)
            {
                var inputTokens = _tokenizer.Encode(conv.Input, addSpecialTokens: true);
                var targetTokens = _tokenizer.Encode(conv.Target, addSpecialTokens: true);

                if (inputTokens.Length > _maxSequenceLength)
                {
                    inputTokens = inputTokens.Take(_maxSequenceLength).ToArray();
                }

                if (targetTokens.Length > _maxSequenceLength)
                {
                    targetTokens = targetTokens.Take(_maxSequenceLength).ToArray();
                }

                inputs.Add(inputTokens);
                targets.Add(targetTokens);
            }

            if (shuffle)
            {
                var shuffled = ShuffleData(inputs, targets);
                return shuffled;
            }

            return (inputs, targets);
        }

        public CausalExample BuildCausalExample(string prompt, string response)
        {
            var sequence = _tokenizer.EncodeCausalSequence(prompt, response);
            int sepIndex = Array.IndexOf(sequence, _tokenizer.SepToken);

            if (sequence.Length > _maxSequenceLength)
            {
                sequence = sequence.Take(_maxSequenceLength).ToArray();
            }

            if (sepIndex < 0 || sepIndex >= sequence.Length - 1)
            {
                return new CausalExample { Tokens = sequence, ResponseStartIndex = -1 };
            }

            return new CausalExample { Tokens = sequence, ResponseStartIndex = sepIndex };
        }

        public List<CausalExample> LoadCausalConversationData(
            List<ConversationPair> conversations,
            bool shuffle = true)
        {
            var examples = new List<CausalExample>();

            foreach (var conversation in conversations)
            {
                var example = BuildCausalExample(conversation.Input, conversation.Target);

                if (example.ResponseStartIndex >= 0)
                {
                    examples.Add(example);
                }
            }

            if (shuffle)
            {
                return ShuffleCausalData(examples);
            }

            return examples;
        }

        public (List<CausalExample> train, List<CausalExample> validation) SplitCausalData(
            List<CausalExample> examples,
            float validationSplit = 0.2f)
        {
            int validationSize = (int)(examples.Count * validationSplit);
            int trainSize = examples.Count - validationSize;

            var train = examples.Take(trainSize).ToList();
            var validation = examples.Skip(trainSize).ToList();

            return (train, validation);
        }

        private List<CausalExample> ShuffleCausalData(List<CausalExample> examples)
        {
            return examples.OrderBy(_ => _shuffleRandom.Next()).ToList();
        }

        public (List<int[]> inputs, List<int[]> targets) LoadFromIndexedJsonFile(string filepath)
        {
            if (!File.Exists(filepath))
            {
                throw new FileNotFoundException($"Dataset file not found: {filepath}");
            }

            var json = File.ReadAllText(filepath);
            var jsonDoc = JsonDocument.Parse(json);

            var inputs = new List<int[]>();
            var targets = new List<int[]>();
            var responses = new List<string>();

            if (jsonDoc.RootElement.TryGetProperty("respuestas", out var responsesElement))
            {
                foreach (var response in responsesElement.EnumerateArray())
                {
                    responses.Add(response.GetString() ?? "");
                }
                Console.WriteLine($"Loaded {responses.Count} response templates");
            }

            if (jsonDoc.RootElement.TryGetProperty("entrenamiento", out var trainingElement))
            {
                int processedCount = 0;
                int skippedCount = 0;

                foreach (var item in trainingElement.EnumerateArray())
                {
                    var inputWords = new List<string>();

                    if (item.TryGetProperty("entrada", out var inputElement))
                    {
                        foreach (var word in inputElement.EnumerateArray())
                        {
                            var wordStr = word.GetString();
                            if (!string.IsNullOrWhiteSpace(wordStr))
                            {
                                inputWords.Add(wordStr);
                            }
                        }
                    }

                    if (inputWords.Count == 0)
                    {
                        skippedCount++;
                        continue;
                    }

                    string targetText = string.Empty;

                    if (item.TryGetProperty("respuesta", out var responseElement))
                    {
                        if (responseElement.ValueKind == JsonValueKind.Array)
                        {
                            var responseWords = new List<string>();
                            foreach (var word in responseElement.EnumerateArray())
                            {
                                var wordStr = word.GetString();
                                if (!string.IsNullOrWhiteSpace(wordStr))
                                {
                                    responseWords.Add(wordStr);
                                }
                            }
                            targetText = string.Join(" ", responseWords);
                        }
                        else if (responseElement.ValueKind == JsonValueKind.Number)
                        {
                            int responseIndex = responseElement.GetInt32();
                            if (responseIndex >= 0 && responseIndex < responses.Count)
                            {
                                targetText = responses[responseIndex];
                            }
                            else
                            {
                                Console.WriteLine($"Warning: Response index {responseIndex} out of range (0-{responses.Count - 1})");
                                skippedCount++;
                                continue;
                            }
                        }
                        else if (responseElement.ValueKind == JsonValueKind.String)
                        {
                            targetText = responseElement.GetString() ?? string.Empty;
                        }
                    }

                    if (string.IsNullOrWhiteSpace(targetText))
                    {
                        skippedCount++;
                        continue;
                    }

                    var inputText = string.Join(" ", inputWords);
                    var inputTokens = _tokenizer.Encode(inputText, addSpecialTokens: true);
                    var targetTokens = _tokenizer.Encode(targetText, addSpecialTokens: true);

                    if (inputTokens.Length > _maxSequenceLength)
                    {
                        inputTokens = inputTokens.Take(_maxSequenceLength).ToArray();
                    }

                    if (targetTokens.Length > _maxSequenceLength)
                    {
                        targetTokens = targetTokens.Take(_maxSequenceLength).ToArray();
                    }

                    inputs.Add(inputTokens);
                    targets.Add(targetTokens);
                    processedCount++;
                }

                Console.WriteLine($"Processed {processedCount} training pairs (skipped {skippedCount})");
            }

            if (inputs.Count == 0)
            {
                throw new InvalidOperationException("No valid training data found in file");
            }

            return ShuffleData(inputs, targets);
        }

        public (List<int[]> inputs, List<int[]> targets) LoadSequenceData(
            List<string> texts,
            bool shuffle = true)
        {
            var inputs = new List<int[]>();
            var targets = new List<int[]>();

            foreach (var text in texts)
            {
                var tokens = _tokenizer.Encode(text, addSpecialTokens: false);

                if (tokens.Length < 2)
                {
                    continue;
                }

                for (int i = 0; i < tokens.Length - 1; i++)
                {
                    int contextLength = Math.Min(i + 1, _maxSequenceLength);
                    var input = tokens.Skip(Math.Max(0, i + 1 - contextLength)).Take(contextLength).ToArray();
                    var target = new int[] { tokens[i + 1] };

                    inputs.Add(input);
                    targets.Add(target);
                }
            }

            if (shuffle)
            {
                var shuffled = ShuffleData(inputs, targets);
                return shuffled;
            }

            return (inputs, targets);
        }

        public (List<int[]> inputs, List<int[]> targets) LoadFromJsonFile(string filepath)
        {
            try
            {
                return LoadFromIndexedJsonFile(filepath);
            }
            catch
            {
                if (!File.Exists(filepath))
                {
                    throw new FileNotFoundException($"Dataset file not found: {filepath}");
                }

                var json = File.ReadAllText(filepath);
                var dataset = JsonSerializer.Deserialize<ConversationDataset>(json);

                if (dataset == null || dataset.Conversations == null)
                {
                    throw new InvalidOperationException("Failed to load dataset from file");
                }

                return LoadConversationData(dataset.Conversations);
            }
        }

        public (List<int[]> inputs, List<int[]> targets) LoadFromTextFile(
            string filepath,
            string separator = "\n")
        {
            if (!File.Exists(filepath))
            {
                throw new FileNotFoundException($"Text file not found: {filepath}");
            }

            var content = File.ReadAllText(filepath);
            var texts = content.Split(separator, StringSplitOptions.RemoveEmptyEntries)
                .Select(t => t.Trim())
                .Where(t => !string.IsNullOrEmpty(t))
                .ToList();

            return LoadSequenceData(texts);
        }

        public (List<int[]> train, List<int[]> validation, List<int[]> trainTargets, List<int[]> validationTargets)
            SplitData(List<int[]> inputs, List<int[]> targets, float validationSplit = 0.2f)
        {
            if (inputs.Count != targets.Count)
            {
                throw new ArgumentException("Inputs and targets must have the same count");
            }

            int validationSize = (int)(inputs.Count * validationSplit);
            int trainSize = inputs.Count - validationSize;

            var trainInputs = inputs.Take(trainSize).ToList();
            var trainTargets = targets.Take(trainSize).ToList();

            var validationInputs = inputs.Skip(trainSize).ToList();
            var validationTargets = targets.Skip(trainSize).ToList();

            return (trainInputs, validationInputs, trainTargets, validationTargets);
        }

        private (List<int[]>, List<int[]>) ShuffleData(List<int[]> inputs, List<int[]> targets)
        {
            var indices = Enumerable.Range(0, inputs.Count).OrderBy(_ => _shuffleRandom.Next()).ToList();

            var shuffledInputs = new List<int[]>();
            var shuffledTargets = new List<int[]>();

            foreach (var index in indices)
            {
                shuffledInputs.Add(inputs[index]);
                shuffledTargets.Add(targets[index]);
            }

            return (shuffledInputs, shuffledTargets);
        }

        public List<int[]> PadBatch(List<int[]> sequences)
        {
            return _tokenizer.PadBatch(sequences, _maxSequenceLength);
        }

        public DatasetStatistics GetStatistics(List<int[]> inputs, List<int[]> targets)
        {
            return new DatasetStatistics
            {
                NumSamples = inputs.Count,
                AvgInputLength = (float)inputs.Average(s => s.Length),
                MaxInputLength = inputs.Max(s => s.Length),
                MinInputLength = inputs.Min(s => s.Length),
                AvgTargetLength = (float)targets.Average(s => s.Length),
                MaxTargetLength = targets.Max(s => s.Length),
                MinTargetLength = targets.Min(s => s.Length),
                VocabSize = _tokenizer.VocabSize
            };
        }

        public void SaveDataset(string filepath, List<ConversationPair> conversations)
        {
            var dataset = new ConversationDataset
            {
                Conversations = conversations
            };

            var json = JsonSerializer.Serialize(dataset, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            File.WriteAllText(filepath, json);
        }

        public static List<ConversationPair> ParsePlainTextConversations(string filepath)
        {
            if (!File.Exists(filepath))
            {
                throw new FileNotFoundException($"Dataset file not found: {filepath}");
            }

            const string userPrefix = "Usuario:";
            const string assistantPrefix = "Asistente:";

            var conversations = new List<ConversationPair>();
            var lines = File.ReadAllLines(filepath);

            string? userText = null;
            string? assistantText = null;
            bool appendingToAssistant = false;

            void FlushPair()
            {
                if (userText != null && assistantText != null)
                {
                    conversations.Add(new ConversationPair
                    {
                        Input = userText.Trim(),
                        Target = assistantText.Trim()
                    });
                }

                userText = null;
                assistantText = null;
                appendingToAssistant = false;
            }

            foreach (var rawLine in lines)
            {
                var line = rawLine.TrimEnd();

                if (string.IsNullOrWhiteSpace(line))
                {
                    FlushPair();
                    continue;
                }

                if (line.StartsWith(userPrefix, StringComparison.OrdinalIgnoreCase))
                {
                    FlushPair();
                    userText = line.Substring(userPrefix.Length).Trim();
                    appendingToAssistant = false;
                    continue;
                }

                if (line.StartsWith(assistantPrefix, StringComparison.OrdinalIgnoreCase))
                {
                    assistantText = line.Substring(assistantPrefix.Length).Trim();
                    appendingToAssistant = true;
                    continue;
                }

                if (appendingToAssistant && assistantText != null)
                {
                    assistantText = assistantText + " " + line.Trim();
                }
                else if (userText != null)
                {
                    userText = userText + " " + line.Trim();
                }
            }

            FlushPair();

            return conversations;
        }

        public static void ConvertIndexedJsonToPlainText(string jsonFilepath, string outputTxtFilepath)
        {
            var conversations = new DatasetLoader(new Tokenizer()).LoadConversationPairs(jsonFilepath);

            using var writer = new StreamWriter(outputTxtFilepath, false, System.Text.Encoding.UTF8);

            foreach (var conversation in conversations)
            {
                writer.WriteLine($"Usuario: {conversation.Input}");
                writer.WriteLine($"Asistente: {conversation.Target}");
                writer.WriteLine();
            }
        }

        public List<ConversationPair> LoadConversationPairs(string filepath)
        {
            if (!File.Exists(filepath))
            {
                throw new FileNotFoundException($"Conversation file not found: {filepath}");
            }

            try
            {
                var json = File.ReadAllText(filepath);
                var jsonDoc = JsonDocument.Parse(json);
                var conversations = new List<ConversationPair>();
                var responses = new List<string>();

                if (jsonDoc.RootElement.TryGetProperty("respuestas", out var responsesElement))
                {
                    foreach (var response in responsesElement.EnumerateArray())
                    {
                        responses.Add(response.GetString() ?? "");
                    }
                }

                if (jsonDoc.RootElement.TryGetProperty("entrenamiento", out var trainingElement))
                {
                    foreach (var item in trainingElement.EnumerateArray())
                    {
                        var inputWords = new List<string>();

                        if (item.TryGetProperty("entrada", out var inputElement))
                        {
                            foreach (var word in inputElement.EnumerateArray())
                            {
                                var wordStr = word.GetString();
                                if (!string.IsNullOrWhiteSpace(wordStr))
                                {
                                    inputWords.Add(wordStr);
                                }
                            }
                        }

                        string targetText = string.Empty;

                        if (item.TryGetProperty("respuesta", out var responseElement))
                        {
                            if (responseElement.ValueKind == JsonValueKind.Array)
                            {
                                var responseWords = new List<string>();
                                foreach (var word in responseElement.EnumerateArray())
                                {
                                    var wordStr = word.GetString();
                                    if (!string.IsNullOrWhiteSpace(wordStr))
                                    {
                                        responseWords.Add(wordStr);
                                    }
                                }
                                targetText = string.Join(" ", responseWords);
                            }
                            else if (responseElement.ValueKind == JsonValueKind.Number)
                            {
                                int responseIndex = responseElement.GetInt32();
                                if (responseIndex >= 0 && responseIndex < responses.Count)
                                {
                                    targetText = responses[responseIndex];
                                }
                            }
                        }

                        if (inputWords.Count > 0 && !string.IsNullOrWhiteSpace(targetText))
                        {
                            conversations.Add(new ConversationPair
                            {
                                Input = string.Join(" ", inputWords),
                                Target = targetText
                            });
                        }
                    }
                }

                return conversations;
            }
            catch
            {
                var json = File.ReadAllText(filepath);
                var dataset = JsonSerializer.Deserialize<ConversationDataset>(json);
                return dataset?.Conversations ?? new List<ConversationPair>();
            }
        }

        public void CreateSyntheticDataset(string outputPath, int numSamples = 1000)
        {
            var conversations = new List<ConversationPair>();
            var random = new Random();

            var greetings = new[] { "hola", "buenos dias", "buenas tardes", "hey", "que tal" };
            var responses = new[] { "hola como estas", "muy bien gracias", "todo bien y tu", "genial" };

            for (int i = 0; i < numSamples; i++)
            {
                var input = greetings[random.Next(greetings.Length)];
                var target = responses[random.Next(responses.Length)];

                conversations.Add(new ConversationPair
                {
                    Input = input,
                    Target = target
                });
            }

            SaveDataset(outputPath, conversations);
            Console.WriteLine($"Synthetic dataset created: {numSamples} samples at {outputPath}");
        }

        public (List<int[]> inputs, List<int[]> targets) AugmentData(
            List<int[]> inputs,
            List<int[]> targets,
            int augmentationFactor = 2)
        {
            var augmentedInputs = new List<int[]>(inputs);
            var augmentedTargets = new List<int[]>(targets);

            var random = new Random();

            for (int factor = 0; factor < augmentationFactor - 1; factor++)
            {
                for (int i = 0; i < inputs.Count; i++)
                {
                    var augmented = AddNoise(inputs[i], random, 0.1f);
                    augmentedInputs.Add(augmented);
                    augmentedTargets.Add(targets[i]);
                }
            }

            return ShuffleData(augmentedInputs, augmentedTargets);
        }

        private int[] AddNoise(int[] sequence, Random random, float noiseRate)
        {
            var noisy = (int[])sequence.Clone();

            for (int i = 0; i < noisy.Length; i++)
            {
                if (random.NextSingle() < noiseRate)
                {
                    noisy[i] = random.Next(_tokenizer.VocabSize);
                }
            }

            return noisy;
        }
    }

    public class CausalExample
    {
        public int[] Tokens { get; set; } = Array.Empty<int>();
        public int ResponseStartIndex { get; set; } = -1;
    }

    public class ConversationPair
    {
        public string Input { get; set; }
        public string Target { get; set; }

        public ConversationPair()
        {
            Input = string.Empty;
            Target = string.Empty;
        }
    }

    public class ConversationDataset
    {
        public List<ConversationPair> Conversations { get; set; }

        public ConversationDataset()
        {
            Conversations = new List<ConversationPair>();
        }
    }

    public class DatasetStatistics
    {
        public int NumSamples { get; set; }
        public float AvgInputLength { get; set; }
        public int MaxInputLength { get; set; }
        public int MinInputLength { get; set; }
        public float AvgTargetLength { get; set; }
        public int MaxTargetLength { get; set; }
        public int MinTargetLength { get; set; }
        public int VocabSize { get; set; }

        public override string ToString()
        {
            return $"Dataset Statistics:\n" +
                   $"  Samples: {NumSamples}\n" +
                   $"  Avg Input Length: {AvgInputLength:F2}\n" +
                   $"  Max Input Length: {MaxInputLength}\n" +
                   $"  Min Input Length: {MinInputLength}\n" +
                   $"  Avg Target Length: {AvgTargetLength:F2}\n" +
                   $"  Max Target Length: {MaxTargetLength}\n" +
                   $"  Min Target Length: {MinTargetLength}\n" +
                   $"  Vocab Size: {VocabSize}";
        }
    }
}