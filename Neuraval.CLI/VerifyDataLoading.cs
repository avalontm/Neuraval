using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Neuraval.Core.Services;
using Neuraval.ChatBot.Services;

namespace Neuraval.CLI
{
    public class DiagnosticTool
    {
        public static void VerifyDataLoading(string jsonPath)
        {
            Console.WriteLine("===========================================");
            Console.WriteLine("         DATA LOADING DIAGNOSTIC");
            Console.WriteLine("===========================================");
            Console.WriteLine();

            try
            {
                if (!File.Exists(jsonPath))
                {
                    Console.WriteLine($"ERROR: File not found: {jsonPath}");
                    return;
                }

                Console.WriteLine($"File: {Path.GetFileName(jsonPath)}");
                Console.WriteLine();

                var jsonContent = File.ReadAllText(jsonPath);
                var jsonDoc = JsonDocument.Parse(jsonContent);

                Console.WriteLine("Step 1: Analyzing JSON structure...");

                if (!jsonDoc.RootElement.TryGetProperty("vocabulario", out var vocabElement))
                {
                    Console.WriteLine("  WARNING: No 'vocabulario' property found");
                }
                else
                {
                    int vocabSize = vocabElement.EnumerateObject().Count();
                    Console.WriteLine($"  Vocabulary size: {vocabSize} tokens");
                }

                var conversations = new List<(string input, string target)>();

                if (jsonDoc.RootElement.TryGetProperty("entrenamiento", out var trainingElement))
                {
                    Console.WriteLine($"  Training entries: {trainingElement.GetArrayLength()}");
                    Console.WriteLine();

                    Console.WriteLine("Step 2: Loading conversation pairs...");

                    foreach (var item in trainingElement.EnumerateArray())
                    {
                        var inputWords = new List<string>();
                        string targetText = string.Empty;

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
                                targetText = $"[Response Index: {responseElement.GetInt32()}]";
                            }
                            else if (responseElement.ValueKind == JsonValueKind.String)
                            {
                                targetText = responseElement.GetString() ?? string.Empty;
                            }
                        }

                        if (inputWords.Count > 0 && !string.IsNullOrWhiteSpace(targetText))
                        {
                            conversations.Add((string.Join(" ", inputWords), targetText));
                        }
                    }

                    Console.WriteLine($"  Loaded {conversations.Count} conversation pairs");
                    Console.WriteLine();
                }
                else
                {
                    Console.WriteLine("  ERROR: No 'entrenamiento' property found");
                    return;
                }

                Console.WriteLine("First 5 conversation pairs:");
                for (int i = 0; i < Math.Min(5, conversations.Count); i++)
                {
                    Console.WriteLine($"  {i + 1}. Input:  \"{conversations[i].input}\"");
                    Console.WriteLine($"     Target: \"{conversations[i].target}\"");
                    Console.WriteLine();
                }

                var tokenizer = new Tokenizer();

                Console.WriteLine("Step 3: Building vocabulary from conversations...");
                var allTexts = conversations
                    .SelectMany(c => new[] { c.input, c.target })
                    .ToList();

                tokenizer.BuildVocabulary(allTexts, minFrequency: 1, maxVocabSize: 10000);
                Console.WriteLine($"  Built vocabulary with {tokenizer.VocabSize} tokens");
                Console.WriteLine();

                var vocab = tokenizer.GetVocabulary();
                Console.WriteLine("Vocabulary sample (first 20 tokens):");
                int count = 0;
                foreach (var kvp in vocab.OrderBy(x => x.Value).Take(20))
                {
                    Console.WriteLine($"  {kvp.Value}: \"{kvp.Key}\"");
                    count++;
                }
                Console.WriteLine();

                Console.WriteLine("Step 4: Testing tokenization...");
                var testInput = conversations.Count > 0 ? conversations[0].input : "0 + 0";
                var testTarget = conversations.Count > 0 ? conversations[0].target : "el resultado es cero";

                var inputTokens = tokenizer.Encode(testInput, addSpecialTokens: true);
                var targetTokens = tokenizer.Encode(testTarget, addSpecialTokens: true);

                Console.WriteLine($"  Input text:  \"{testInput}\"");
                Console.WriteLine($"  Input tokens: [{string.Join(", ", inputTokens)}]");
                Console.WriteLine($"  Decoded back: \"{tokenizer.Decode(inputTokens, skipSpecialTokens: true)}\"");
                Console.WriteLine();

                Console.WriteLine($"  Target text:  \"{testTarget}\"");
                Console.WriteLine($"  Target tokens: [{string.Join(", ", targetTokens)}]");
                Console.WriteLine($"  Target length: {targetTokens.Length} tokens");
                Console.WriteLine($"  Decoded back: \"{tokenizer.Decode(targetTokens, skipSpecialTokens: true)}\"");
                Console.WriteLine();

                Console.WriteLine("Step 5: Loading with DatasetLoader...");
                var dataLoader = new DatasetLoader(tokenizer, 128);

                try
                {
                    var (inputs, targets) = dataLoader.LoadFromIndexedJsonFile(jsonPath);

                    Console.WriteLine($"  Total training pairs: {inputs.Count}");
                    Console.WriteLine($"  Average input length: {inputs.Average(x => x.Length):F1} tokens");
                    Console.WriteLine($"  Average target length: {targets.Average(x => x.Length):F1} tokens");
                    Console.WriteLine();

                    Console.WriteLine("First 3 training pairs (tokenized):");
                    for (int i = 0; i < Math.Min(3, inputs.Count); i++)
                    {
                        var inputText = tokenizer.Decode(inputs[i], skipSpecialTokens: false);
                        var targetText = tokenizer.Decode(targets[i], skipSpecialTokens: false);

                        Console.WriteLine($"  Pair {i + 1}:");
                        Console.WriteLine($"    Input tokens:  [{string.Join(", ", inputs[i])}]");
                        Console.WriteLine($"    Input decoded: \"{inputText}\"");
                        Console.WriteLine($"    Target tokens: [{string.Join(", ", targets[i])}]");
                        Console.WriteLine($"    Target decoded: \"{targetText}\"");
                        Console.WriteLine($"    Target length: {targets[i].Length} tokens");
                        Console.WriteLine();
                    }

                    Console.WriteLine("===========================================");
                    Console.WriteLine("         VALIDATION CHECKS");
                    Console.WriteLine("===========================================");
                    Console.WriteLine();

                    bool allGood = true;

                    if (tokenizer.VocabSize < 50)
                    {
                        Console.WriteLine($"  WARNING: Vocabulary small ({tokenizer.VocabSize} tokens)");
                        Console.WriteLine("           This is OK for arithmetic, but may limit responses");
                    }
                    else
                    {
                        Console.WriteLine($"  PASS: Vocabulary size OK ({tokenizer.VocabSize} tokens)");
                    }

                    var avgTargetLength = targets.Average(x => x.Length);
                    if (avgTargetLength < 3)
                    {
                        Console.WriteLine($"  FAIL: Target length too short ({avgTargetLength:F1} tokens)");
                        allGood = false;
                    }
                    else
                    {
                        Console.WriteLine($"  PASS: Target length OK ({avgTargetLength:F1} tokens average)");
                    }

                    var requiredWords = new[] { "el", "resultado", "es" };
                    var missingWords = requiredWords.Where(w => !tokenizer.ContainsToken(w)).ToList();

                    if (missingWords.Any())
                    {
                        Console.WriteLine($"  WARNING: Missing important words: {string.Join(", ", missingWords)}");
                    }
                    else
                    {
                        Console.WriteLine("  PASS: Important response words in vocabulary");
                    }

                    if (inputs.Count == 0)
                    {
                        Console.WriteLine("  FAIL: No training data loaded");
                        allGood = false;
                    }
                    else
                    {
                        Console.WriteLine($"  PASS: {inputs.Count} training pairs loaded successfully");
                    }

                    Console.WriteLine();

                    if (allGood)
                    {
                        Console.WriteLine("  ALL CHECKS PASSED - Data is ready for training!");
                    }
                    else
                    {
                        Console.WriteLine("  SOME CHECKS FAILED - Review issues above");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  ERROR loading with DatasetLoader: {ex.Message}");
                    Console.WriteLine();
                    Console.WriteLine("  This means the DatasetLoader needs the updated code.");
                    Console.WriteLine("  Make sure you have replaced DatasetLoader.cs with the fixed version.");
                }

                Console.WriteLine();
                Console.WriteLine("===========================================");
                Console.WriteLine("         DIAGNOSTIC COMPLETE");
                Console.WriteLine("===========================================");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"ERROR: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}