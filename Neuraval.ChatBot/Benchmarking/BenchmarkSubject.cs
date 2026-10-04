using System;
using Neuraval.Abstractions;

namespace Neuraval.ChatBot.Benchmarking
{
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
