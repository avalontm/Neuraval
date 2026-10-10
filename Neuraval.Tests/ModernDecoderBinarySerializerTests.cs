using Neuraval.Core.Models;
using Neuraval.Core.Serialization;
using Neuraval.Core.Tokenizers;
using Neuraval.ChatBot.Services;
using Xunit;

namespace Neuraval.Tests;

public sealed class ModernDecoderBinarySerializerTests
{
    [Fact]
    public void SaveAndLoad_RoundTripsModernDecoderWeightsTokenizerAndTemplate()
    {
        string path = Path.Combine(Path.GetTempPath(), $"neuraval-{Guid.NewGuid():N}.navm");
        var config = new TransformerConfig
        {
            Architecture = "test",
            VocabSize = 7,
            HiddenSize = 4,
            NumHiddenLayers = 1,
            NumAttentionHeads = 2,
            NumKeyValueHeads = 1,
            IntermediateSize = 8,
            MaxPositionEmbeddings = 16,
            TieWordEmbeddings = true
        };
        var model = new ModernDecoderModel(config, seed: 7);
        var tokenizer = new ModernBpeTokenizer();
        var template = ChatTemplateDefinition.ChatMl();
        int[] prompt = { tokenizer.StartToken, tokenizer.GetTokenId("<|im_start|>") };

        try
        {
            float[,] before = model.Forward(prompt);
            ModernDecoderBinarySerializer.Save(path, model.SaveState(), tokenizer, template);

            Assert.True(ModernDecoderBinarySerializer.HasModernSignature(path));
            Assert.False(ModelBinarySerializer.IsNavmFile(path));
            var restored = ModernDecoderBinarySerializer.Load(path);
            float[,] after = restored.Model.Forward(prompt);
            var chatModel = GgufChatModel.FromFile(path);

            Assert.Equal(before.GetLength(0), after.GetLength(0));
            Assert.Equal(before.GetLength(1), after.GetLength(1));
            for (int row = 0; row < before.GetLength(0); row++)
                for (int column = 0; column < before.GetLength(1); column++)
                    Assert.InRange(Math.Abs(before[row, column] - after[row, column]), 0f, 1e-2f);

            Assert.Equal(tokenizer.VocabSize, restored.Tokenizer.VocabSize);
            Assert.Equal(template.Name, restored.ChatTemplate.Name);
            Assert.NotNull(chatModel);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
