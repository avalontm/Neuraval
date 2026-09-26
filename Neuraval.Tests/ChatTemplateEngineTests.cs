using Neuraval.Abstractions;
using Neuraval.Core.Tokenizers;
using Xunit;

namespace Neuraval.Tests
{
    public class ChatTemplateEngineTests
    {
        [Fact]
        public void Render_WrapsEachMessageWithImStartAndImEnd()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.System, "You are a helpful assistant."),
                new ChatMessage(ChatRole.User, "Hola")
            };

            var rendered = ChatTemplateEngine.Render(messages, addGenerationPrompt: false);

            Assert.Contains("<|im_start|>system", rendered);
            Assert.Contains("You are a helpful assistant.", rendered);
            Assert.Contains("<|im_start|>user", rendered);
            Assert.Contains("Hola", rendered);
            Assert.Contains("<|im_end|>", rendered);
        }

        [Fact]
        public void Render_AddGenerationPrompt_AppendsAssistantOpenTag()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hola")
            };

            var rendered = ChatTemplateEngine.Render(messages, addGenerationPrompt: true);

            Assert.EndsWith("<|im_start|>assistant\n", rendered);
        }

        [Fact]
        public void Render_WithoutGenerationPrompt_DoesNotAppendAssistantOpenTag()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hola")
            };

            var rendered = ChatTemplateEngine.Render(messages, addGenerationPrompt: false);

            Assert.DoesNotContain("<|im_start|>assistant", rendered);
        }

        [Fact]
        public void Render_PreservesMessageOrder()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.System, "system-content"),
                new ChatMessage(ChatRole.User, "user-content"),
                new ChatMessage(ChatRole.Assistant, "assistant-content")
            };

            var rendered = ChatTemplateEngine.Render(messages, addGenerationPrompt: false);

            int systemIndex = rendered.IndexOf("system-content", StringComparison.Ordinal);
            int userIndex = rendered.IndexOf("user-content", StringComparison.Ordinal);
            int assistantIndex = rendered.IndexOf("assistant-content", StringComparison.Ordinal);

            Assert.True(systemIndex < userIndex);
            Assert.True(userIndex < assistantIndex);
        }

        [Fact]
        public void Render_ToolRole_UsesToolRoleName()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.Tool, "tool-result")
            };

            var rendered = ChatTemplateEngine.Render(messages, addGenerationPrompt: false);

            Assert.Contains("<|im_start|>tool", rendered);
            Assert.Contains("tool-result", rendered);
        }

        [Fact]
        public void Render_PlainTemplate_UsesConfiguredFormatInsteadOfChatMl()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hola")
            };

            var rendered = ChatTemplateEngine.Render(messages, ChatTemplateDefinition.Plain(), addGenerationPrompt: false);

            Assert.Contains("### user:\nHola", rendered);
            Assert.DoesNotContain("<|im_start|>", rendered);
        }

        [Fact]
        public void Render_DifferentTemplates_ProduceDifferentOutputForSameMessages()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hola")
            };

            var chatMl = ChatTemplateEngine.Render(messages, ChatTemplateDefinition.ChatMl(), addGenerationPrompt: false);
            var plain = ChatTemplateEngine.Render(messages, ChatTemplateDefinition.Plain(), addGenerationPrompt: false);

            Assert.NotEqual(chatMl, plain);
        }

        [Fact]
        public void Render_MistralTemplate_WrapsUserMessageInInstTags()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hola")
            };

            var rendered = ChatTemplateEngine.Render(messages, ChatTemplateDefinition.Mistral(), addGenerationPrompt: false);

            Assert.Equal("[INST] Hola [/INST]", rendered);
        }

        [Fact]
        public void Render_MistralTemplate_MergesSystemMessageIntoFirstUserTurn()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.System, "Sos un asistente útil."),
                new ChatMessage(ChatRole.User, "Hola")
            };

            var rendered = ChatTemplateEngine.Render(messages, ChatTemplateDefinition.Mistral(), addGenerationPrompt: false);

            // No debe aparecer como turno propio: Mistral clásico no tiene rol "system".
            Assert.DoesNotContain("[INST] Sos un asistente útil. [/INST]", rendered);
            Assert.Contains("Sos un asistente útil.", rendered);
            Assert.Contains("Hola", rendered);
            Assert.StartsWith("[INST] ", rendered);
        }

        [Fact]
        public void Render_MistralTemplate_AssistantTurnEndsWithEosTag()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hola"),
                new ChatMessage(ChatRole.Assistant, "¡Hola! ¿En qué te ayudo?")
            };

            var rendered = ChatTemplateEngine.Render(messages, ChatTemplateDefinition.Mistral(), addGenerationPrompt: false);

            Assert.EndsWith("</s>", rendered);
        }

        [Fact]
        public void Render_MistralTemplate_GenerationPromptDoesNotAppendRoleTag()
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hola")
            };

            var withPrompt = ChatTemplateEngine.Render(messages, ChatTemplateDefinition.Mistral(), addGenerationPrompt: true);
            var withoutPrompt = ChatTemplateEngine.Render(messages, ChatTemplateDefinition.Mistral(), addGenerationPrompt: false);

            // "[/INST]" ya invita al modelo a continuar; a diferencia de ChatML,
            // Mistral no agrega ningún tag extra de rol para el turno del assistant.
            Assert.Equal(withPrompt, withoutPrompt);
        }
    }
}
