using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace Neuraval.Abstractions
{
    public interface IChatModel
    {
        Task<ChatMessage> SendAsync(IReadOnlyList<ChatMessage> messages, CancellationToken cancellationToken = default);
    }
}
