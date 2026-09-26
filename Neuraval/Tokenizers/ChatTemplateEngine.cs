using Neuraval.Abstractions;

namespace Neuraval.Core.Tokenizers
{
    public static class ChatTemplateEngine
    {
        public static string Render(IReadOnlyList<ChatMessage> messages, ChatTemplateDefinition? definition = null, bool addGenerationPrompt = true)
        {
            if (messages == null)
                throw new ArgumentNullException(nameof(messages));

            definition ??= ChatTemplateDefinition.ChatMl();

            // Si el GGUF trae su propio chat_template Jinja (tokenizer.chat_template),
            // ese es el formato exacto con el que se entrenó/fine-tuneó el modelo, así
            // que tiene prioridad total sobre los presets heurísticos de abajo.
            if (definition.RawJinjaTemplate != null)
            {
                return JinjaChatTemplateEngine.Render(
                    definition.RawJinjaTemplate,
                    messages,
                    addGenerationPrompt,
                    definition.BosToken ?? string.Empty,
                    definition.EosToken ?? string.Empty);
            }

            // Los formatos "por rol" (p.ej. Mistral, con "[INST] ... [/INST]") no
            // envuelven cada turno igual según el rol, así que necesitan su propio
            // camino de render en vez del esquema genérico de ChatML de abajo.
            return definition.RoleTemplates != null
                ? RenderWithRoleTemplates(messages, definition, addGenerationPrompt)
                : RenderGeneric(messages, definition, addGenerationPrompt);
        }

        private static string RenderGeneric(IReadOnlyList<ChatMessage> messages, ChatTemplateDefinition definition, bool addGenerationPrompt)
        {
            var builder = new System.Text.StringBuilder();

            foreach (var message in messages)
            {
                builder.Append(definition.TurnStart);
                builder.Append(definition.RoleName(message.Role));
                builder.Append(definition.RoleHeaderSeparator);
                builder.Append(message.Content);
                builder.Append(definition.TurnEnd);
            }

            if (addGenerationPrompt)
            {
                builder.Append(definition.TurnStart);
                builder.Append(definition.GenerationPromptRole);
                builder.Append(definition.RoleHeaderSeparator);
            }

            return builder.ToString();
        }

        private static string RenderWithRoleTemplates(IReadOnlyList<ChatMessage> messages, ChatTemplateDefinition definition, bool addGenerationPrompt)
        {
            // El llamador (Render) ya garantizó que RoleTemplates no es null antes
            // de entrar acá; se guarda en una variable local no-nula para que el
            // resto del método pueda usarlo sin advertencias de nulabilidad.
            var roleTemplates = definition.RoleTemplates!;

            var builder = new System.Text.StringBuilder();
            bool systemPending = definition.MergeSystemIntoFirstUser;
            string? pendingSystemContent = null;

            foreach (var message in messages)
            {
                // Formatos como Mistral clásico no tienen turno de "system" propio:
                // se guarda el contenido y se antepone al primer mensaje de usuario.
                if (systemPending && message.Role == ChatRole.System)
                {
                    pendingSystemContent = message.Content;
                    continue;
                }

                if (!roleTemplates.TryGetValue(message.Role, out var roleTemplate))
                    throw new ArgumentOutOfRangeException(
                        nameof(messages),
                        $"El template '{definition.Name}' no define un formato para el rol {message.Role}");

                var content = message.Content;

                if (message.Role == ChatRole.User && pendingSystemContent != null)
                {
                    content = pendingSystemContent + "\n\n" + content;
                    pendingSystemContent = null;
                }

                builder.Append(roleTemplate.Prefix);
                builder.Append(content);
                builder.Append(roleTemplate.Suffix);
            }

            if (addGenerationPrompt)
                builder.Append(definition.GenerationPromptText);

            return builder.ToString();
        }
    }
}
