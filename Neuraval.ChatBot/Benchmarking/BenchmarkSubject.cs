using System;
using Neuraval.Abstractions;

namespace Neuraval.ChatBot.Benchmarking
{
    /// <summary>
    /// Representa un backend de chat a comparar (Neuraval Native, Qwen, Gemma, Llama, GPT/API, ...).
    /// La metadata estática (cantidad de parámetros, tamaño en disco) es opcional porque no todos los
    /// backends la exponen: un endpoint remoto tipo GPT/API no permite inspeccionar esos datos desde acá,
    /// así que quedan en null en vez de inventarse.
    /// </summary>
    public sealed class BenchmarkSubject
    {
        public string Name { get; }
        public IChatModel Model { get; }
        public ITokenCounter TokenCounter { get; }
        public long? ParameterCount { get; }
        public long? ModelSizeBytes { get; }

        public BenchmarkSubject(
            string name,
            IChatModel model,
            ITokenCounter? tokenCounter = null,
            long? parameterCount = null,
            long? modelSizeBytes = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("name is required", nameof(name));

            Name = name;
            Model = model ?? throw new ArgumentNullException(nameof(model));
            TokenCounter = tokenCounter ?? new WhitespaceTokenCounter();
            ParameterCount = parameterCount;
            ModelSizeBytes = modelSizeBytes;
        }
    }
}
