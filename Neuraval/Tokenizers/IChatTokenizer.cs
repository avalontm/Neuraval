using System.Collections.Generic;
using Neuraval.Abstractions;
using Neuraval.Core.Services;

namespace Neuraval.Core.Tokenizers
{
    // Extiende ITokenizer con lo que GgufChatModel necesita además de Encode/Decode:
    // plantilla de chat y los tokens de turno (im_start/im_end) usados para detener la
    // generación. La implementan tanto ModernBpeTokenizer (BPE byte-level estilo gpt2) como
    // SentencePieceBpeTokenizer (tokenizer.ggml.model = "llama").
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
