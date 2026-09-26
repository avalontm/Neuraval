using Neuraval.Abstractions;
using Neuraval.Core.Tokenizers;
using Xunit;

namespace Neuraval.Tests
{
    public class ChatTemplateDefinitionTests
    {
        [Fact]
        public void ChatMl_HasExpectedDefaults()
        {
            var definition = ChatTemplateDefinition.ChatMl();

            Assert.Equal("chatml", definition.Name);
            Assert.Equal("<|im_start|>", definition.TurnStart);
            Assert.Equal("assistant", definition.GenerationPromptRole);
        }

        [Fact]
        public void Plain_HasExpectedDefaults()
        {
            var definition = ChatTemplateDefinition.Plain();

            Assert.Equal("plain", definition.Name);
            Assert.Equal("### ", definition.TurnStart);
            Assert.Equal("assistant", definition.GenerationPromptRole);
        }

        [Fact]
        public void RoleName_ReturnsConfiguredNameForEachRole()
        {
            var definition = ChatTemplateDefinition.ChatMl();

            Assert.Equal("system", definition.RoleName(ChatRole.System));
            Assert.Equal("user", definition.RoleName(ChatRole.User));
            Assert.Equal("assistant", definition.RoleName(ChatRole.Assistant));
            Assert.Equal("tool", definition.RoleName(ChatRole.Tool));
        }

        [Fact]
        public void RoleName_CustomTemplate_UsesOverriddenNames()
        {
            var definition = ChatTemplateDefinition.ChatMl();
            definition.RoleNames[ChatRole.User] = "human";

            Assert.Equal("human", definition.RoleName(ChatRole.User));
        }

        [Fact]
        public void RoleName_MissingRoleMapping_Throws()
        {
            var definition = ChatTemplateDefinition.ChatMl();
            definition.RoleNames.Remove(ChatRole.Tool);

            Assert.Throws<ArgumentOutOfRangeException>(() => definition.RoleName(ChatRole.Tool));
        }

        [Fact]
        public void Mistral_HasExpectedDefaults()
        {
            var definition = ChatTemplateDefinition.Mistral();

            Assert.Equal("mistral", definition.Name);
            Assert.True(definition.MergeSystemIntoFirstUser);
            Assert.NotNull(definition.RoleTemplates);
            Assert.Equal("[INST] ", definition.RoleTemplates![ChatRole.User].Prefix);
            Assert.Equal(" [/INST]", definition.RoleTemplates[ChatRole.User].Suffix);
            Assert.Equal("</s>", definition.RoleTemplates[ChatRole.Assistant].Suffix);
        }
    }
}
