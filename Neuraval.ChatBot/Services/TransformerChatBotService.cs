using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Neuraval.Abstractions;
using Neuraval.Core.Models;
using Neuraval.Core.Serialization;
using Neuraval.Core.Services;
using Neuraval.Core.Utils;

namespace Neuraval.ChatBot.Services
{
    public class TransformerChatBotService : IChatModel
    {
        private TransformerModel? _model;
        private ITokenizer _tokenizer;
        private bool _isTrained;

        private int _embeddingDim;
        private int _numLayers;
        private int _numHeads;
        private int _feedforwardDim;
        private int _maxSequenceLength;
        private readonly double _dropout;
        private int _numThreads;

        public TransformerChatBotService(
            int embeddingDim = 128,
            int numLayers = 4,
            int numHeads = 4,
            int feedforwardDim = 512,
            int maxSequenceLength = 128,
            double dropout = 0.1,
            int numThreads = -1)
        {
            _embeddingDim = embeddingDim;
            _numLayers = numLayers;
            _numHeads = numHeads;
            _feedforwardDim = feedforwardDim;
            _maxSequenceLength = maxSequenceLength;
            _dropout = dropout;
            _numThreads = numThreads == -1 ? Environment.ProcessorCount : numThreads;

            Matematicas.SetNumThreads(_numThreads);

            _tokenizer = new BpeTokenizer();
            _isTrained = false;

            Console.WriteLine("Parallelization configured:");
            Console.WriteLine($"  Number of threads: {_numThreads}");
            Console.WriteLine($"  Processor count: {Environment.ProcessorCount}");
        }

        public void BuildVocabularyFromTexts(List<string> texts, int minFrequency = 1, int maxVocabSize = 10000)
        {
            if (_tokenizer is Tokenizer wordLevelTokenizer)
            {
                wordLevelTokenizer.BuildVocabulary(texts, minFrequency, maxVocabSize);
            }
            else
            {
                _tokenizer.BuildVocabulary(texts, maxVocabSize);
            }

            _model = new TransformerModel(
                vocabSize: _tokenizer.VocabSize,
                embeddingDim: _embeddingDim,
                numLayers: _numLayers,
                numHeads: _numHeads,
                feedforwardDim: _feedforwardDim,
                maxSequenceLength: _maxSequenceLength,
                dropout: (float)_dropout);

            Console.WriteLine("Transformer model initialized:");
            Console.WriteLine($"  Vocabulary Size: {_tokenizer.VocabSize} (target: {maxVocabSize})");
            Console.WriteLine($"  Embedding table: {_tokenizer.VocabSize}x{_embeddingDim} (~{_tokenizer.VocabSize * (long)_embeddingDim * sizeof(float) / (1024.0 * 1024.0):F1} MB, embedding + output projection)");
            Console.WriteLine($"  Embedding Dimension: {_embeddingDim}");
            Console.WriteLine($"  Number of Layers: {_numLayers}");
            Console.WriteLine($"  Number of Heads: {_numHeads}");
            Console.WriteLine($"  Max Sequence Length: {_maxSequenceLength}");
        }

        public void TrainWithConversations(
            List<ConversationPair> conversations,
            int epochs = 100,
            int batchSize = 32,
            double learningRate = 0.001,
            double validationSplit = 0.2,
            int patience = 10,
            Action<int, double, double>? onEpochCompleted = null,
            string? checkpointFolder = null,
            int checkpointEveryEpochs = 0,
            string? logFilePath = null,
            TrainingProgressState? resumeFrom = null,
            int gradientAccumulationSteps = 1)
        {
            if (_model == null || _tokenizer == null)
            {
                throw new InvalidOperationException("Model not initialized. Call BuildVocabularyFromTexts first.");
            }

            _isTrained = true;

            var dataLoader = new DatasetLoader(_tokenizer, _maxSequenceLength);
            var causalExamples = dataLoader.LoadCausalConversationData(conversations);
            var (trainExamples, validationExamples) = dataLoader.SplitCausalData(causalExamples, (float)validationSplit);

            Console.WriteLine();
            Console.WriteLine("Training dataset:");
            Console.WriteLine($"  Training samples: {trainExamples.Count}");
            Console.WriteLine($"  Validation samples: {validationExamples.Count}");
            Console.WriteLine();

            var trainer = new SupervisedTrainer(_model, (float)learningRate, padToken: _tokenizer.PadToken);

            Action<TrainingProgressState>? onCheckpoint = null;
            if (checkpointFolder != null)
            {
                onCheckpoint = progress =>
                {
                    SaveCompleteModel(checkpointFolder);
                    SaveTrainingProgress(checkpointFolder, progress);
                    Console.WriteLine($"Checkpoint saved at epoch {progress.LastCompletedEpoch} to: {checkpointFolder}");
                };
            }

            Action<int, float, float>? onEpochCompletedAdapter = onEpochCompleted == null
                ? null
                : (epoch, trainLoss, valLoss) => onEpochCompleted(epoch, trainLoss, valLoss);

            ITrainer<TransformerModel, CausalTrainingDataset> universalTrainer = trainer;
            universalTrainer.Train(_model, new CausalTrainingDataset
            {
                TrainingExamples = trainExamples,
                ValidationExamples = validationExamples,
                Epochs = epochs,
                BatchSize = batchSize,
                Patience = patience,
                OnEpochCompleted = onEpochCompletedAdapter,
                ResumeFrom = resumeFrom,
                CheckpointIntervalEpochs = checkpointEveryEpochs,
                OnCheckpoint = onCheckpoint,
                LogFilePath = logFilePath,
                GradientAccumulationSteps = gradientAccumulationSteps
            });
        }

        public void TrainWithTexts(
            List<string> texts,
            int epochs = 100,
            int batchSize = 32,
            double learningRate = 0.001,
            double validationSplit = 0.2)
        {
            if (_model == null || _tokenizer == null)
            {
                throw new InvalidOperationException("Model not initialized. Call BuildVocabularyFromTexts first.");
            }

            _isTrained = true;

            var dataLoader = new DatasetLoader(_tokenizer, _maxSequenceLength);
            var (inputs, targets) = dataLoader.LoadSequenceData(texts);

            var (trainInputs, valInputs, trainTargets, valTargets) =
                dataLoader.SplitData(inputs, targets, (float)validationSplit);

            Console.WriteLine();
            Console.WriteLine("Training dataset:");
            Console.WriteLine($"  Training samples: {trainInputs.Count}");
            Console.WriteLine($"  Validation samples: {valInputs.Count}");

            var trainer = new SupervisedTrainer(_model, (float)learningRate, padToken: _tokenizer.PadToken);

            if (valInputs.Count > 0)
            {
                trainer.TrainWithValidation(
                    trainInputs, trainTargets,
                    valInputs, valTargets,
                    epochs, batchSize);
            }
            else
            {
                trainer.Train(trainInputs, trainTargets, epochs, batchSize);
            }

            Console.WriteLine();
            Console.WriteLine("Training completed!");
        }

        public string GenerateResponse(string userMessage, int maxLength = 50)
        {
            if (!_isTrained)
            {
                return "The chatbot has not been trained yet.";
            }

            if (string.IsNullOrWhiteSpace(userMessage))
            {
                return "I didn't understand that. Could you please rephrase?";
            }

            try
            {
                var inputTokens = _tokenizer.EncodePrompt(userMessage);

                int adaptiveMaxLength = Math.Min(maxLength, 15);

                var generatedTokens = GenerateGreedy(inputTokens, adaptiveMaxLength);
                var response = _tokenizer.Decode(generatedTokens.ToArray(), skipSpecialTokens: true);

                if (string.IsNullOrWhiteSpace(response))
                {
                    return "I'm not sure how to respond to that.";
                }

                return response.Trim();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating response: {ex.Message}");
                return "Sorry, I encountered an error generating a response.";
            }
        }

        public Task<ChatMessage> SendAsync(IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            ChatMessage? lastUserMessage = null;
            for (int i = messages.Count - 1; i >= 0; i--)
            {
                if (messages[i].Role == ChatRole.User)
                {
                    lastUserMessage = messages[i];
                    break;
                }
            }

            if (lastUserMessage == null)
            {
                throw new ChatModelException("The conversation does not contain a user message.");
            }

            return Task.Run(() =>
            {
                var responseText = GenerateResponse(lastUserMessage.Content);
                return new ChatMessage(ChatRole.Assistant, responseText);
            }, cancellationToken);
        }

        private List<int> GenerateGreedy(int[] inputTokens, int maxLength)
        {
            var currentSequence = new List<int>(inputTokens);
            var generatedTokens = new List<int>();

            var recentTokens = new Dictionary<int, int>();
            int consecutiveRepeats = 0;
            int lastToken = -1;

            for (int i = 0; i < maxLength; i++)
            {
                var sequenceToUse = currentSequence.Count > _maxSequenceLength
                    ? currentSequence.Skip(currentSequence.Count - _maxSequenceLength).ToArray()
                    : currentSequence.ToArray();

                var probabilities = _model!.Predict(sequenceToUse);

                var candidates = probabilities
                    .Select((prob, idx) => new { Prob = prob, Index = idx })
                    .OrderByDescending(x => x.Prob)
                    .ToList();

                int nextToken = -1;

                foreach (var candidate in candidates)
                {
                    int token = candidate.Index;

                    if (token == _tokenizer.PadToken || token == _tokenizer.UnknownToken)
                        continue;

                    if (token == _tokenizer.EndToken)
                    {
                        nextToken = token;
                        break;
                    }

                    if (token == lastToken && consecutiveRepeats >= 2)
                        continue;

                    if (recentTokens.TryGetValue(token, out int lastPos))
                    {
                        if (i - lastPos < 5 && candidate.Prob < 0.3)
                            continue;
                    }

                    nextToken = token;
                    break;
                }

                if (nextToken == -1)
                {
                    foreach (var candidate in candidates)
                    {
                        if (candidate.Index != _tokenizer.PadToken &&
                            candidate.Index != _tokenizer.UnknownToken &&
                            candidate.Index != _tokenizer.StartToken)
                        {
                            nextToken = candidate.Index;
                            break;
                        }
                    }
                }

                if (nextToken == -1)
                    nextToken = ArgMax(probabilities);

                if (nextToken == _tokenizer.EndToken)
                    break;

                if (nextToken == lastToken)
                    consecutiveRepeats++;
                else
                    consecutiveRepeats = 0;

                if (consecutiveRepeats >= 3)
                    break;

                generatedTokens.Add(nextToken);
                currentSequence.Add(nextToken);
                recentTokens[nextToken] = i;
                lastToken = nextToken;
            }

            return generatedTokens;
        }

        public string GenerateWithTemperature(
            string userMessage,
            int maxLength = 50,
            double temperature = 1.0)
        {
            if (!_isTrained)
            {
                return "The chatbot has not been trained yet.";
            }

            if (string.IsNullOrWhiteSpace(userMessage))
            {
                return "I didn't understand that. Could you please rephrase?";
            }

            try
            {
                var inputTokens = _tokenizer.EncodePrompt(userMessage);

                int adaptiveMaxLength = Math.Min(maxLength, 15);

                var generatedTokens = GenerateWithTemperatureSampling(inputTokens, adaptiveMaxLength, temperature);
                var response = _tokenizer.Decode(generatedTokens.ToArray(), skipSpecialTokens: true);

                if (string.IsNullOrWhiteSpace(response))
                {
                    return "I'm not sure how to respond to that.";
                }

                return response.Trim();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating response: {ex.Message}");
                return "Sorry, I encountered an error generating a response.";
            }
        }

        private List<int> GenerateWithTemperatureSampling(
            int[] inputTokens,
            int maxLength,
            double temperature)
        {
            var currentSequence = new List<int>(inputTokens);
            var generatedTokens = new List<int>();
            var random = new Random();

            var recentTokens = new List<int>();
            int repetitionWindow = 5;

            for (int i = 0; i < maxLength; i++)
            {
                var sequenceToUse = currentSequence.Count > _maxSequenceLength
                    ? currentSequence.Skip(currentSequence.Count - _maxSequenceLength).ToArray()
                    : currentSequence.ToArray();

                var logits = _model!.Forward(sequenceToUse, false);
                int lastPosition = logits.GetLength(0) - 1;

                var lastLogits = new float[_model.VocabSize];
                for (int j = 0; j < _model.VocabSize; j++)
                {
                    lastLogits[j] = logits[lastPosition, j];
                }

                foreach (var recentToken in recentTokens)
                {
                    lastLogits[recentToken] -= 2.0f;
                }

                var scaledLogits = ApplyTemperature(lastLogits, temperature);
                var probabilities = Matematicas.ParallelSoftmax(scaledLogits);

                probabilities[_tokenizer.PadToken] = 0.0f;
                probabilities[_tokenizer.UnknownToken] = 0.0f;

                var sum = probabilities.Sum();
                if (sum > 0)
                {
                    for (int j = 0; j < probabilities.Length; j++)
                    {
                        probabilities[j] /= sum;
                    }
                }

                var nextToken = SampleFromDistribution(probabilities, random);

                if (nextToken == _tokenizer.EndToken)
                {
                    break;
                }

                generatedTokens.Add(nextToken);
                currentSequence.Add(nextToken);

                recentTokens.Add(nextToken);
                if (recentTokens.Count > repetitionWindow)
                {
                    recentTokens.RemoveAt(0);
                }
            }

            return generatedTokens;
        }

        public string GenerateWithBeamSearch(
            string userMessage,
            int maxLength = 50,
            int beamWidth = 5)
        {
            if (!_isTrained)
            {
                return "The chatbot has not been trained yet.";
            }

            try
            {
                var inputTokens = _tokenizer.EncodePrompt(userMessage);

                int adaptiveMaxLength = Math.Min(maxLength, 15);

                var generatedTokens = BeamSearch(inputTokens, adaptiveMaxLength, beamWidth);
                var response = _tokenizer.Decode(generatedTokens.ToArray(), skipSpecialTokens: true);

                return string.IsNullOrWhiteSpace(response)
                    ? "I'm not sure how to respond to that."
                    : response.Trim();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating response: {ex.Message}");
                return "Sorry, I encountered an error generating a response.";
            }
        }

        private List<int> BeamSearch(int[] inputTokens, int maxLength, int beamWidth)
        {
            var beams = new List<Beam>
            {
                new Beam
                {
                    Tokens = new List<int>(inputTokens),
                    Score = 0.0,
                    RecentTokens = new HashSet<int>()
                }
            };

            for (int step = 0; step < maxLength; step++)
            {
                var candidates = new List<Beam>();

                foreach (var beam in beams)
                {
                    if (beam.Tokens.Count > inputTokens.Length)
                    {
                        var lastToken = beam.Tokens[beam.Tokens.Count - 1];
                        if (lastToken == _tokenizer.EndToken)
                        {
                            candidates.Add(beam);
                            continue;
                        }
                    }

                    var sequenceToUse = beam.Tokens.Count > _maxSequenceLength
                        ? beam.Tokens.Skip(beam.Tokens.Count - _maxSequenceLength).ToArray()
                        : beam.Tokens.ToArray();

                    var probabilities = _model!.Predict(sequenceToUse);

                    foreach (var recentToken in beam.RecentTokens)
                    {
                        probabilities[recentToken] *= 0.5f;
                    }

                    probabilities[_tokenizer.PadToken] = 0.0f;
                    probabilities[_tokenizer.UnknownToken] = 0.0f;

                    var topK = GetTopK(probabilities, beamWidth);

                    foreach (var (token, prob) in topK)
                    {
                        if (prob < 1e-10) continue;

                        var newTokens = new List<int>(beam.Tokens) { token };
                        var newScore = beam.Score + Math.Log(prob + 1e-10);

                        var newRecentTokens = new HashSet<int>(beam.RecentTokens) { token };
                        if (newRecentTokens.Count > 5)
                        {
                            newRecentTokens.Remove(newRecentTokens.First());
                        }

                        candidates.Add(new Beam
                        {
                            Tokens = newTokens,
                            Score = newScore,
                            RecentTokens = newRecentTokens
                        });
                    }
                }

                if (candidates.Count == 0)
                {
                    break;
                }

                beams = candidates
                    .OrderByDescending(b =>
                    {
                        var generatedLength = Math.Max(1, b.Tokens.Count - inputTokens.Length);
                        return b.Score / Math.Pow(generatedLength, 0.7);
                    })
                    .Take(beamWidth)
                    .ToList();

                bool allFinished = beams.All(b =>
                {
                    if (b.Tokens.Count <= inputTokens.Length) return false;
                    var lastToken = b.Tokens[b.Tokens.Count - 1];
                    return lastToken == _tokenizer.EndToken;
                });

                if (allFinished)
                {
                    break;
                }
            }

            var bestBeam = beams.OrderByDescending(b =>
            {
                var generatedLength = Math.Max(1, b.Tokens.Count - inputTokens.Length);
                return b.Score / Math.Pow(generatedLength, 0.7);
            }).First();

            return bestBeam.Tokens.Skip(inputTokens.Length).ToList();
        }

        public List<string> GenerateMultipleCandidates(
            string userMessage,
            int numCandidates = 5,
            int maxLength = 50,
            double temperature = 0.8)
        {
            if (!_isTrained)
            {
                return new List<string> { "The chatbot has not been trained yet." };
            }

            try
            {
                var responses = new HashSet<string>();
                int attempts = 0;
                int maxAttempts = numCandidates * 3;

                while (responses.Count < numCandidates && attempts < maxAttempts)
                {
                    attempts++;
                    var response = GenerateWithTemperature(userMessage, maxLength, temperature);

                    if (!string.IsNullOrWhiteSpace(response) &&
                        !response.Contains("error") &&
                        !response.Contains("Error"))
                    {
                        responses.Add(response);
                    }
                }

                return responses.Any() ? responses.ToList() : new List<string> { "Could not generate responses." };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error generating multiple responses: {ex.Message}");
                return new List<string> { "Sorry, I encountered an error." };
            }
        }

        private class Beam
        {
            public List<int> Tokens { get; set; } = new List<int>();
            public double Score { get; set; }
            public HashSet<int> RecentTokens { get; set; } = new HashSet<int>();
        }

        private int ArgMax(float[] array)
        {
            int maxIndex = 0;
            float maxValue = array[0];

            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] > maxValue)
                {
                    maxValue = array[i];
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        private float[] ApplyTemperature(float[] logits, double temperature)
        {
            var result = new float[logits.Length];

            for (int i = 0; i < logits.Length; i++)
            {
                result[i] = (float)(logits[i] / temperature);
            }

            return result;
        }

        private int SampleFromDistribution(float[] probabilities, Random random)
        {
            var r = random.NextDouble();
            var cumulative = 0.0;

            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (r <= cumulative)
                    return i;
            }

            return probabilities.Length - 1;
        }

        private List<(int token, float prob)> GetTopK(float[] probabilities, int k)
        {
            return probabilities
                .Select((prob, index) => (token: index, prob: prob))
                .OrderByDescending(x => x.prob)
                .Take(k)
                .ToList();
        }

        public bool IsTrained()
        {
            return _isTrained;
        }

        public int GetVocabularySize()
        {
            return _tokenizer?.VocabSize ?? 0;
        }

        private const string BinaryModelFileName = "model" + ModelBinaryFormat.FileExtension;

        public void SaveCompleteModel(string destinationFolder)
        {
            if (_model == null || _tokenizer == null)
            {
                throw new InvalidOperationException("Model not initialized");
            }

            if (!Directory.Exists(destinationFolder))
            {
                Directory.CreateDirectory(destinationFolder);
            }

            var tokenizerPath = Path.Combine(destinationFolder, "tokenizer.json");
            _tokenizer.SaveToFile(tokenizerPath);

            var modelState = _model.SaveState();
            var header = new ModelBinaryHeader
            {
                VocabSize = _tokenizer.VocabSize,
                EmbeddingDim = _embeddingDim,
                NumLayers = _numLayers,
                NumHeads = _numHeads,
                FeedforwardDim = _feedforwardDim,
                MaxSequenceLength = _maxSequenceLength,
                NumThreads = _numThreads,
                IsTrained = _isTrained,
                TokenizerType = _tokenizer is BpeTokenizer ? "bpe" : "wordlevel"
            };

            var modelPath = Path.Combine(destinationFolder, BinaryModelFileName);
            ModelBinarySerializer.Save(modelPath, modelState, header);

            Console.WriteLine($"Model saved successfully to: {destinationFolder}");
            Console.WriteLine($"  Format: {ModelBinaryFormat.FileExtension} (binary, GZip-compressed)");
        }

        public bool LoadCompleteModel(string sourceFolder)
        {
            try
            {
                var tokenizerPath = Path.Combine(sourceFolder, "tokenizer.json");
                var binaryModelPath = Path.Combine(sourceFolder, BinaryModelFileName);

                if (!File.Exists(tokenizerPath) || !File.Exists(binaryModelPath))
                {
                    Console.WriteLine("Missing required files in model directory");
                    return false;
                }

                var (modelState, header) = ModelBinarySerializer.Load(binaryModelPath);

                _tokenizer = header.TokenizerType == "bpe"
                    ? BpeTokenizer.LoadFromFile(tokenizerPath)
                    : Tokenizer.LoadFromFile(tokenizerPath);

                if (header.NumThreads > 0)
                {
                    Matematicas.SetNumThreads(header.NumThreads);
                }

                _model = TransformerModel.LoadState(modelState);
                _isTrained = header.IsTrained;

                _embeddingDim = header.EmbeddingDim;
                _numLayers = header.NumLayers;
                _numHeads = header.NumHeads;
                _feedforwardDim = header.FeedforwardDim;
                _maxSequenceLength = header.MaxSequenceLength;
                if (header.NumThreads > 0)
                {
                    _numThreads = header.NumThreads;
                }

                Console.WriteLine($"Model loaded successfully from: {sourceFolder}");
                Console.WriteLine($"  Tokenizer: {(header.TokenizerType == "bpe" ? "BPE" : "word-level")}");
                Console.WriteLine($"  Vocabulary Size: {_tokenizer.VocabSize}");
                Console.WriteLine($"  Model trained: {_isTrained}");
                Console.WriteLine($"  Threads: {Matematicas.GetNumThreads()}");

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error loading model: {ex.Message}");
                return false;
            }
        }

        private static void SaveTrainingProgress(string folder, TrainingProgressState progress)
        {
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }

            var path = Path.Combine(folder, "training_progress.json");
            var json = System.Text.Json.JsonSerializer.Serialize(progress,
                new System.Text.Json.JsonSerializerOptions
                {
                    WriteIndented = true
                });
            File.WriteAllText(path, json);
        }

        public static TrainingProgressState? LoadTrainingProgress(string folder)
        {
            var path = Path.Combine(folder, "training_progress.json");

            if (!File.Exists(path))
            {
                return null;
            }

            var json = File.ReadAllText(path);
            return System.Text.Json.JsonSerializer.Deserialize<TrainingProgressState>(json);
        }

        public static void DeleteTrainingProgress(string folder)
        {
            var path = Path.Combine(folder, "training_progress.json");

            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }

        public void LoadConversationsFromJson(string filepath)
        {
            var dataLoader = new DatasetLoader(_tokenizer, _maxSequenceLength);
            var conversations = dataLoader.LoadConversationPairs(filepath);

            Console.WriteLine($"Loaded {conversations.Count} conversation pairs from {filepath}");
        }

        public ITokenizer GetTokenizer()
        {
            return _tokenizer;
        }

        public TransformerModel? GetModel()
        {
            return _model;
        }
    }
}
