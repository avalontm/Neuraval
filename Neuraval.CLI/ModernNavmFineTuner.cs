using System.Globalization;
using System.Text.Json;
using Neuraval.Abstractions;
using Neuraval.Core.Serialization;

namespace Neuraval.CLI;

internal static class ModernNavmFineTuner
{
    public static void Run(string[] args)
    {
        if (args.Length < 4)
        {
            PrintUsage();
            return;
        }

        string basePath = Path.GetFullPath(args[1]);
        string dataPath = Path.GetFullPath(args[2]);
        string outputPath = Path.GetFullPath(args[3]);
        string? validationPathOption = Option(args, "--validation-jsonl");
        string? validationPath = validationPathOption == null ? null : Path.GetFullPath(validationPathOption);
        int epochs = ReadInt(args, "--epochs", 1, min: 1, max: 100);
        int requestedMaxSequenceLength = ReadInt(args, "--max-seq-len", 256, min: 2, max: 1_000_000);
        float learningRate = ReadFloat(args, "--learning-rate", 0.00001f);

        if (!File.Exists(basePath) || !ModernDecoderBinarySerializer.HasModernSignature(basePath))
            throw new ArgumentException($"El modelo base no es un .navm moderno válido: {basePath}");
        if (!File.Exists(dataPath))
            throw new FileNotFoundException("No se encontró el dataset JSONL.", dataPath);
        if (validationPath != null && !File.Exists(validationPath))
            throw new FileNotFoundException("No se encontró el dataset JSONL de validación.", validationPath);
        if (string.Equals(basePath, outputPath, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("La salida debe ser otro archivo; se protege el modelo base original.");

        var examples = LoadExamples(dataPath);
        if (examples.Count == 0)
            throw new InvalidDataException("El dataset no contiene conversaciones válidas.");

        Console.WriteLine($"Cargando modelo entrenable: {basePath}");
        Console.WriteLine("Aviso: el ajuste completo convierte Q8 a float32 y reserva gradientes y optimizadores; puede requerir varias veces la RAM del archivo.");
        var file = ModernDecoderBinarySerializer.Load(basePath, inferenceOnly: false);
        var model = file.Model;
        int maxSequenceLength = Math.Min(requestedMaxSequenceLength, model.MaxPositionEmbeddings);
        if (maxSequenceLength < requestedMaxSequenceLength)
            Console.WriteLine($"Longitud limitada por el modelo a {maxSequenceLength} tokens.");

        var tokenized = Tokenize(examples, file, maxSequenceLength, dataPath);
        var validationExamples = validationPath == null ? new List<TrainingExample>() : LoadExamples(validationPath);
        var validation = validationPath == null
            ? new List<(int[] Tokens, int LossStart)>()
            : Tokenize(validationExamples, file, maxSequenceLength, validationPath);

        Console.WriteLine($"Dataset: {tokenized.Count} ejemplos" + (validation.Count == 0 ? string.Empty : $", validación: {validation.Count}") + $"; máximo {maxSequenceLength} tokens; épocas {epochs}; tasa {learningRate.ToString("G", CultureInfo.InvariantCulture)}.");
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            double totalLoss = 0;
            foreach (var item in tokenized)
            {
                model.ResetGradients();
                totalLoss += model.CalculateCausalLoss(item.Tokens, item.LossStart);
                model.UpdateWeights(learningRate);
            }

            float averageLoss = (float)(totalLoss / tokenized.Count);
            float validationLoss = validation.Count == 0 ? float.NaN : Evaluate(model, validation);
            string epochPath = GetEpochPath(outputPath, epoch, epochs);
            ModernDecoderBinarySerializer.Save(epochPath, model.SaveState(), file.Tokenizer, file.ChatTemplate);
            string validationReport = validation.Count == 0 ? string.Empty : $"; val_loss={validationLoss:F5}";
            Console.WriteLine($"[época {epoch}/{epochs}] loss={averageLoss:F5}{validationReport}; guardado: {epochPath}");
        }

        Console.WriteLine("Ajuste terminado. El modelo base permanece intacto; prueba la salida con el comando chat habitual.");
    }

    private static List<(int[] Tokens, int LossStart)> Tokenize(
        IReadOnlyList<TrainingExample> examples,
        ModernDecoderFile file,
        int maxSequenceLength,
        string sourcePath)
    {
        var tokenized = new List<(int[] Tokens, int LossStart)>(examples.Count);
        foreach (var example in examples)
        {
            var promptMessages = example.Messages.Take(example.Messages.Count - 1).ToArray();
            int[] prompt = file.Tokenizer.EncodeChat(promptMessages, file.ChatTemplate, addGenerationPrompt: true);
            int[] sequence = file.Tokenizer.EncodeChat(example.Messages, file.ChatTemplate, addGenerationPrompt: false);
            int lossStart = prompt.Length - 1;
            if (sequence.Length < 2 || lossStart < 0 || lossStart >= sequence.Length - 1)
                throw new InvalidDataException($"{sourcePath}, línea {example.LineNumber}: no produce tokens de respuesta entrenables.");
            if (sequence.Length > maxSequenceLength)
                throw new InvalidDataException($"{sourcePath}, línea {example.LineNumber}: {sequence.Length} tokens exceden --max-seq-len {maxSequenceLength}. Acorta el ejemplo o aumenta el límite.");
            tokenized.Add((sequence, lossStart));
        }
        return tokenized;
    }

    private static float Evaluate(Neuraval.Core.Models.ModernDecoderModel model, IReadOnlyList<(int[] Tokens, int LossStart)> examples)
    {
        double total = 0;
        foreach (var item in examples)
        {
            model.ResetGradients();
            total += model.CalculateCausalLoss(item.Tokens, item.LossStart);
        }
        model.ResetGradients();
        return (float)(total / examples.Count);
    }

    private static List<TrainingExample> LoadExamples(string path)
    {
        var result = new List<TrainingExample>();
        int lineNumber = 0;
        foreach (string line in File.ReadLines(path))
        {
            lineNumber++;
            if (string.IsNullOrWhiteSpace(line)) continue;
            using var document = JsonDocument.Parse(line);
            if (!document.RootElement.TryGetProperty("messages", out var messagesElement) || messagesElement.ValueKind != JsonValueKind.Array)
                throw new InvalidDataException($"Línea {lineNumber}: se esperaba {{\"messages\":[...]}}.");

            var messages = new List<ChatMessage>();
            foreach (var messageElement in messagesElement.EnumerateArray())
            {
                if (!messageElement.TryGetProperty("role", out var roleElement) || !messageElement.TryGetProperty("content", out var contentElement))
                    throw new InvalidDataException($"Línea {lineNumber}: cada mensaje requiere role y content.");
                string roleName = roleElement.GetString() ?? string.Empty;
                string content = contentElement.GetString() ?? string.Empty;
                ChatRole role = roleName.ToLowerInvariant() switch
                {
                    "system" => ChatRole.System,
                    "user" => ChatRole.User,
                    "assistant" => ChatRole.Assistant,
                    _ => throw new InvalidDataException($"Línea {lineNumber}: rol no admitido '{roleName}'.")
                };
                messages.Add(new ChatMessage(role, content));
            }

            if (messages.Count < 2 || messages[^1].Role != ChatRole.Assistant || string.IsNullOrWhiteSpace(messages[^1].Content))
                throw new InvalidDataException($"Línea {lineNumber}: termina con una respuesta assistant no vacía y debe incluir al menos un mensaje previo.");
            result.Add(new TrainingExample(lineNumber, messages));
        }
        return result;
    }

    private static int ReadInt(string[] args, string name, int fallback, int min, int max)
    {
        string? value = Option(args, name);
        if (value == null) return fallback;
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed) || parsed < min || parsed > max)
            throw new ArgumentException($"{name} debe estar entre {min} y {max}.");
        return parsed;
    }

    private static float ReadFloat(string[] args, string name, float fallback)
    {
        string? value = Option(args, name);
        if (value == null) return fallback;
        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed) || !float.IsFinite(parsed) || parsed <= 0 || parsed > 1)
            throw new ArgumentException($"{name} debe ser un número mayor que 0 y hasta 1.");
        return parsed;
    }

    private static string? Option(string[] args, string name)
    {
        int index = Array.FindIndex(args, item => string.Equals(item, name, StringComparison.OrdinalIgnoreCase));
        if (index < 0) return null;
        if (index + 1 >= args.Length || args[index + 1].StartsWith("--", StringComparison.Ordinal))
            throw new ArgumentException($"Falta el valor de {name}.");
        return args[index + 1];
    }

    private static string GetEpochPath(string outputPath, int epoch, int epochs)
    {
        if (epoch == epochs) return outputPath;
        string directory = Path.GetDirectoryName(outputPath)!;
        string name = Path.GetFileNameWithoutExtension(outputPath);
        return Path.Combine(directory, $"{name}.epoch-{epoch:D2}.navm");
    }

    private static void PrintUsage()
    {
        Console.WriteLine("Uso: dotnet run --project Neuraval.CLI -- --finetune-navm <base.navm> <datos.jsonl> <salida.navm> [--validation-jsonl eval.jsonl] [--epochs N] [--learning-rate X] [--max-seq-len N]");
        Console.WriteLine("JSONL: cada línea contiene {\"messages\":[{\"role\":\"user\",\"content\":\"...\"},{\"role\":\"assistant\",\"content\":\"...\"}]}.");
    }

    private sealed record TrainingExample(int LineNumber, List<ChatMessage> Messages);
}
