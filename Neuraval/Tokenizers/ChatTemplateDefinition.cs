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

        public Dictionary<ChatRole, ChatRoleTemplate>? RoleTemplates { get; set; }

        public bool MergeSystemIntoFirstUser { get; set; }

        public string GenerationPromptText { get; set; } = string.Empty;

        public string? RawJinjaTemplate { get; set; }

        public string? BosToken { get; set; }

        public string? EosToken { get; set; }

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
