using Neuraval.Abstractions;

namespace Neuraval.Core.Tokenizers
{
    public sealed class ChatTemplateDefinition
    {
        public string Name { get; set; } = "chatml";

        public string TurnStart { get; set; } = "<|im_start|>";

        public string RoleHeaderSeparator { get; set; } = "\n";

        public string TurnEnd { get; set; } = "\n<|im_end|>\n";

        public string GenerationPromptRole { get; set; } = "assistant";

        public Dictionary<ChatRole, string> RoleNames { get; set; } = DefaultRoleNames();

        /// <summary>
        /// Cuando no es null, el render usa este esquema "por rol" (prefijo/sufijo
        /// propio por mensaje) en vez del esquema genérico TurnStart/RoleName/
        /// RoleHeaderSeparator/TurnEnd de arriba. Necesario para formatos como el
        /// de Mistral, que no envuelve cada turno de la misma forma según el rol.
        /// </summary>
        public Dictionary<ChatRole, ChatRoleTemplate>? RoleTemplates { get; set; }

        /// <summary>
        /// Solo aplica cuando RoleTemplates no es null. Algunos formatos (Mistral
        /// clásico) no tienen un rol "system" propio: el contenido del primer
        /// mensaje de sistema se antepone al primer mensaje de usuario en vez de
        /// renderizarse como un turno separado.
        /// </summary>
        public bool MergeSystemIntoFirstUser { get; set; }

        /// <summary>
        /// Solo aplica cuando RoleTemplates no es null. Texto que se agrega al
        /// final cuando addGenerationPrompt es true. En Mistral queda vacío,
        /// porque "[/INST]" ya funciona como invitación a que el modelo continúe;
        /// en ChatML en cambio hace falta el tag "&lt;|im_start|&gt;assistant\n".
        /// </summary>
        public string GenerationPromptText { get; set; } = string.Empty;

        /// <summary>
        /// Cuando no es null, contiene el chat_template Jinja tal cual lo trae
        /// el metadato "tokenizer.chat_template" del GGUF. Si está presente,
        /// tiene prioridad total sobre TurnStart/RoleTemplates/etc: se renderiza
        /// ejecutando el template real del modelo (ver <see cref="JinjaChatTemplateEngine"/>)
        /// en vez de una aproximación heurística por arquitectura.
        /// </summary>
        public string? RawJinjaTemplate { get; set; }

        /// <summary>
        /// Representación literal (string) del token BOS del vocabulario, para
        /// que el template pueda referenciar "{{ bos_token }}". Solo se usa
        /// junto con RawJinjaTemplate.
        /// </summary>
        public string? BosToken { get; set; }

        /// <summary>
        /// Representación literal (string) del token EOS del vocabulario, para
        /// "{{ eos_token }}". Solo se usa junto con RawJinjaTemplate.
        /// </summary>
        public string? EosToken { get; set; }

        /// <summary>
        /// Construye una definición que renderiza usando el chat_template Jinja
        /// real de un GGUF, en vez de los presets ChatMl/Plain/Mistral.
        /// </summary>
        public static ChatTemplateDefinition FromRawJinja(string jinjaTemplate, string bosToken, string eosToken)
        {
            if (string.IsNullOrEmpty(jinjaTemplate))
                throw new ArgumentException("jinjaTemplate no puede ser vacío", nameof(jinjaTemplate));

            return new ChatTemplateDefinition
            {
                Name = "gguf-jinja",
                RawJinjaTemplate = jinjaTemplate,
                BosToken = bosToken ?? string.Empty,
                EosToken = eosToken ?? string.Empty
            };
        }

        private static Dictionary<ChatRole, string> DefaultRoleNames()
        {
            return new Dictionary<ChatRole, string>
            {
                [ChatRole.System] = "system",
                [ChatRole.User] = "user",
                [ChatRole.Assistant] = "assistant",
                [ChatRole.Tool] = "tool"
            };
        }

        public string RoleName(ChatRole role)
        {
            if (RoleNames.TryGetValue(role, out var name))
                return name;

            throw new ArgumentOutOfRangeException(nameof(role), $"El template '{Name}' no define un nombre para el rol {role}");
        }

        public static ChatTemplateDefinition ChatMl()
        {
            return new ChatTemplateDefinition();
        }

        public static ChatTemplateDefinition Plain()
        {
            return new ChatTemplateDefinition
            {
                Name = "plain",
                TurnStart = "### ",
                RoleHeaderSeparator = ":\n",
                TurnEnd = "\n\n",
                RoleNames = DefaultRoleNames()
            };
        }

        /// <summary>
        /// Template de instrucción estilo Mistral/Llama: "[INST] {user} [/INST]",
        /// la respuesta del assistant termina en "&lt;/s&gt;" y no hay tag de rol
        /// visible. El mensaje de sistema (si hay uno) se antepone al primer
        /// mensaje de usuario porque este formato clásico no tiene un rol
        /// "system" propio. No soporta mensajes de tipo Tool.
        /// </summary>
        public static ChatTemplateDefinition Mistral()
        {
            return new ChatTemplateDefinition
            {
                Name = "mistral",
                MergeSystemIntoFirstUser = true,
                GenerationPromptText = string.Empty,
                RoleTemplates = new Dictionary<ChatRole, ChatRoleTemplate>
                {
                    [ChatRole.User] = new ChatRoleTemplate { Prefix = "[INST] ", Suffix = " [/INST]" },
                    [ChatRole.Assistant] = new ChatRoleTemplate { Prefix = string.Empty, Suffix = "</s>" }
                }
            };
        }
    }
}
