using Neuraval.Abstractions;
using Neuraval.Core.Tokenizers;
using Xunit;

namespace Neuraval.Tests
{
    public class JinjaChatTemplateEngineTests
    {
        [Fact]
        public void Render_ChatMlStyleTemplate_MatchesQwenFormat()
        {
            const string template =
                "{% for message in messages %}" +
                "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}" +
                "{% endfor %}" +
                "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

            var messages = new List<ChatMessage>
            {
                new(ChatRole.System, "Sos un asistente útil."),
                new(ChatRole.User, "hola")
            };

            var rendered = JinjaChatTemplateEngine.Render(template, messages, addGenerationPrompt: true, bosToken: string.Empty, eosToken: "<|im_end|>");

            Assert.Equal(
                "<|im_start|>system\nSos un asistente útil.<|im_end|>\n" +
                "<|im_start|>user\nhola<|im_end|>\n" +
                "<|im_start|>assistant\n",
                rendered);
        }

        [Fact]
        public void Render_MistralInstructTemplate_WrapsUserInInstAndMergesSystem()
        {
            const string template =
                "{{ bos_token }}" +
                "{%- set ns = namespace(system_prompt='') -%}" +
                "{%- for message in messages -%}" +
                "{%- if message['role'] == 'system' -%}" +
                "{%- set ns.system_prompt = message['content'] -%}" +
                "{%- elif message['role'] == 'user' -%}" +
                "{%- if ns.system_prompt -%}" +
                "{{ '[INST] ' + ns.system_prompt + '\n\n' + message['content'] + ' [/INST]' }}" +
                "{%- set ns.system_prompt = '' -%}" +
                "{%- else -%}" +
                "{{ '[INST] ' + message['content'] + ' [/INST]' }}" +
                "{%- endif -%}" +
                "{%- elif message['role'] == 'assistant' -%}" +
                "{{ message['content'] + eos_token }}" +
                "{%- endif -%}" +
                "{%- endfor -%}";

            var messages = new List<ChatMessage>
            {
                new(ChatRole.System, "Sé breve."),
                new(ChatRole.User, "hola"),
                new(ChatRole.Assistant, "¡Hola!"),
                new(ChatRole.User, "¿cómo estás?")
            };

            var rendered = JinjaChatTemplateEngine.Render(template, messages, addGenerationPrompt: false, bosToken: "<s>", eosToken: "</s>");

            Assert.Equal(
                "<s>[INST] Sé breve.\n\nhola [/INST]¡Hola!</s>[INST] ¿cómo estás? [/INST]",
                rendered);
        }

        [Fact]
        public void Render_TernaryAndSlice_SelectsSystemMessageSeparately()
        {
            const string template =
                "{% if messages[0]['role'] == 'system' %}" +
                "{{ 'SYS:' + messages[0]['content'] + '|' }}" +
                "{% set rest = messages[1:] %}" +
                "{% else %}" +
                "{% set rest = messages %}" +
                "{% endif %}" +
                "{% for m in rest %}{{ m['role'] + ':' + m['content'] + ';' }}{% endfor %}";

            var messages = new List<ChatMessage>
            {
                new(ChatRole.System, "reglas"),
                new(ChatRole.User, "hola"),
                new(ChatRole.Assistant, "hey")
            };

            var rendered = JinjaChatTemplateEngine.Render(template, messages, addGenerationPrompt: false, bosToken: string.Empty, eosToken: string.Empty);

            Assert.Equal("SYS:reglas|user:hola;assistant:hey;", rendered);
        }

        [Fact]
        public void Render_WhitespaceControlDefaults_TrimBlocksAndLstripBlocks()
        {
            const string template =
                "{% for message in messages %}\n" +
                "{{ message['content'] }}\n" +
                "{% endfor %}";

            var messages = new List<ChatMessage>
            {
                new(ChatRole.User, "a"),
                new(ChatRole.User, "b")
            };

            var rendered = JinjaChatTemplateEngine.Render(template, messages, addGenerationPrompt: false, bosToken: string.Empty, eosToken: string.Empty);

            Assert.Equal("a\nb\n", rendered);
        }

        [Fact]
        public void Render_UnsupportedToolFilter_ThrowsJinjaTemplateException()
        {
            const string template = "{{ messages | selectattr('role', 'equalto', 'user') | list }}";

            var messages = new List<ChatMessage> { new(ChatRole.User, "hola") };

            Assert.Throws<JinjaTemplateException>(() =>
                JinjaChatTemplateEngine.Render(template, messages, addGenerationPrompt: false, bosToken: string.Empty, eosToken: string.Empty));
        }

        [Fact]
        public void Render_UnknownMessageAttribute_IsUndefinedNotError()
        {
            const string template =
                "{% for message in messages %}" +
                "{% if message.tool_calls is defined %}TOOLS{% else %}{{ message['content'] }}{% endif %}" +
                "{% endfor %}";

            var messages = new List<ChatMessage> { new(ChatRole.User, "hola") };

            var rendered = JinjaChatTemplateEngine.Render(template, messages, addGenerationPrompt: false, bosToken: string.Empty, eosToken: string.Empty);

            Assert.Equal("hola", rendered);
        }
    }
}
