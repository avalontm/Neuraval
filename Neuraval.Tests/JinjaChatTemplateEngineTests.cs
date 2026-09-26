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
            // Reproduce (simplificado) el chat_template real que traen los GGUF de la
            // familia Qwen/ChatML: "{{ bos_token }}" queda vacío porque estos modelos
            // no usan BOS, y cada turno se envuelve en <|im_start|>/<|im_end|>.
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
            // Aproximación del chat_template real de Mistral-Instruct: el system se
            // funde en el primer turno de usuario, y bos_token/eos_token vienen del
            // vocabulario real del GGUF en vez de estar hardcodeados en C#.
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
            // Sin trim_blocks/lstrip_blocks (el comportamiento que usa HuggingFace por
            // defecto al renderizar chat_template) esto quedaría lleno de saltos de
            // línea sobrantes entre cada turno.
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
            // "tojson"/"selectattr"/etc (tool calling) están deliberadamente fuera de
            // alcance: deben fallar de forma clara para que el llamador pueda caer al
            // preset heurístico en vez de producir un render silenciosamente incorrecto.
            const string template = "{{ messages | selectattr('role', 'equalto', 'user') | list }}";

            var messages = new List<ChatMessage> { new(ChatRole.User, "hola") };

            Assert.Throws<JinjaTemplateException>(() =>
                JinjaChatTemplateEngine.Render(template, messages, addGenerationPrompt: false, bosToken: string.Empty, eosToken: string.Empty));
        }

        [Fact]
        public void Render_UnknownMessageAttribute_IsUndefinedNotError()
        {
            // ChatMessage no tiene "tool_calls"/"name": deben leerse como Undefined
            // (falsy) para que las ramas de tool-calling de un template real se
            // salteen solas, en vez de tirar una excepción.
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
