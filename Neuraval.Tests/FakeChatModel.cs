using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Neuraval.Abstractions;

namespace Neuraval.Tests
{
    internal sealed class FakeChatModel : IChatModel
    {
        private readonly Queue<Func<ChatMessage>> _responses;
        public int CallCount { get; private set; }
        public TimeSpan SimulatedDelay { get; set; } = TimeSpan.Zero;

        public FakeChatModel(params Func<ChatMessage>[] responses)
        {
            _responses = new Queue<Func<ChatMessage>>(responses);
        }

        public static FakeChatModel WithFixedReply(string content)
        {
            return new FakeChatModel(() => new ChatMessage(ChatRole.Assistant, content));
        }

        public static FakeChatModel ThatThrows(string errorMessage)
        {
            return new FakeChatModel(() => throw new ChatModelException(errorMessage));
        }

        public async Task<ChatMessage> SendAsync(IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default)
        {
            CallCount++;

            if (SimulatedDelay > TimeSpan.Zero)
                await Task.Delay(SimulatedDelay, cancellationToken).ConfigureAwait(false);

            var next = _responses.Count > 1 ? _responses.Dequeue() : _responses.Peek();
            return next();
        }
    }
}
