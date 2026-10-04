using System.Collections.Generic;
using Neuraval.Abstractions;
using Neuraval.Core.Services;

namespace Neuraval.Core.Tokenizers
{
    public interface IChatTokenizer : ITokenizer
    {
        int ImStartToken { get; }
        int ImEndToken { get; }

        int[] EncodeChat(IReadOnlyList<ChatMessage> messages, ChatTemplateDefinition? template = null, bool addGenerationPrompt = true);

        bool ContainsToken(string token);
        int GetTokenId(string token);
        string GetToken(int id);
        Dictionary<string, int> GetVocabulary();
    }
}
