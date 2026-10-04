using Neuraval.Abstractions;
using Neuraval.Core.Tokenizers;
using Xunit;

namespace Neuraval.Tests
{
    public class ModernBpeTokenizerTests
    {
        private static List<string> SampleCorpus()
        {
            return new List<string>
            {
                "Hello world, this is a test.",
                "Hello there, world!",
                "Testing the tokenizer with Some MIXED Case text.",
                "Numbers like 123 and 456 should also work.",
                "áéíóú ñ unicode text works too."
            };
        }

        [Fact]
        public void BuildVocabulary_RegistersAllBaseByteSymbols()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            Assert.True(tokenizer.VocabSize >= 256 + 7);
        }

        [Fact]
        public void Encode_DoesNotLowercaseInput()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            var lowerIds = tokenizer.Encode("mixed", addSpecialTokens: false);
            var upperIds = tokenizer.Encode("MIXED", addSpecialTokens: false);

            Assert.NotEqual(lowerIds, upperIds);
        }

        [Fact]
        public void EncodeDecode_RoundTrips_PreservesCaseAndPunctuation()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            const string text = "Hello World, Testing 123!";
            var ids = tokenizer.Encode(text, addSpecialTokens: false);
            var decoded = tokenizer.Decode(ids, skipSpecialTokens: false);

            Assert.Equal(text, decoded);
        }

        [Fact]
        public void EncodeDecode_UnseenText_NeverFallsBackToUnknown()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            var ids = tokenizer.Encode("Completely unseen sentence, ZORB!", addSpecialTokens: false);

            Assert.DoesNotContain(tokenizer.UnknownToken, ids);
        }

        [Fact]
        public void Encode_AddsStartAndEndTokens()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            var ids = tokenizer.Encode("Hello world");

            Assert.Equal(tokenizer.StartToken, ids[0]);
            Assert.Equal(tokenizer.EndToken, ids[^1]);
        }

        [Fact]
        public void EncodeCausalSequence_UsesSepBetweenPromptAndResponse()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            var ids = tokenizer.EncodeCausalSequence("Hello", "world");

            Assert.Equal(tokenizer.StartToken, ids[0]);
            Assert.Contains(tokenizer.SepToken, ids);
            Assert.Equal(tokenizer.EndToken, ids[^1]);
        }

        [Fact]
        public void BuildVocabulary_WithTargetVocabSize_DoesNotExceedTarget()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus(), vocabSize: 300);

            Assert.True(tokenizer.VocabSize <= 300);
        }

        [Fact]
        public void PadSequence_PadsWithPadToken()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            var padded = tokenizer.PadSequence(new[] { 1, 2, 3 }, maxLength: 6);

            Assert.Equal(6, padded.Length);
            Assert.Equal(tokenizer.PadToken, padded[5]);
        }

        [Fact]
        public void EncodeChat_WrapsMessagesWithImStartAndImEnd()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hello world")
            };

            var ids = tokenizer.EncodeChat(messages, addGenerationPrompt: true);

            Assert.Contains(tokenizer.ImStartToken, ids);
            Assert.Contains(tokenizer.ImEndToken, ids);
        }

        [Fact]
        public void EncodeChat_StartsWithBosToken()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hello world")
            };

            var ids = tokenizer.EncodeChat(messages, addGenerationPrompt: true);

            Assert.Equal(tokenizer.StartToken, ids[0]);
        }

        [Fact]
        public void EncodeChat_WithPlainTemplate_DoesNotContainImStartToken()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            var messages = new List<ChatMessage>
            {
                new ChatMessage(ChatRole.User, "Hello world")
            };

            var ids = tokenizer.EncodeChat(messages, ChatTemplateDefinition.Plain(), addGenerationPrompt: false);
            var decoded = tokenizer.Decode(ids, skipSpecialTokens: false);

            Assert.DoesNotContain("<|im_start|>", decoded);
            Assert.Contains("### user:\nHello world", decoded);
        }

        [Fact]
        public void SaveState_LoadState_ProducesIdenticalEncoding()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            const string text = "Hello World, Testing 123!";
            var expected = tokenizer.Encode(text);

            var reloaded = ModernBpeTokenizer.LoadState(tokenizer.SaveState());
            var actual = reloaded.Encode(text);

            Assert.Equal(expected, actual);
        }

        [Fact]
        public void SaveToFile_LoadFromFile_ProducesIdenticalEncoding()
        {
            var tokenizer = new ModernBpeTokenizer();
            tokenizer.BuildVocabulary(SampleCorpus());

            var path = Path.Combine(Path.GetTempPath(), $"modern-bpe-{Guid.NewGuid():N}.json");

            try
            {
                tokenizer.SaveToFile(path);
                var reloaded = ModernBpeTokenizer.LoadFromFile(path);

                const string text = "Hello World, Testing 123!";
                Assert.Equal(tokenizer.Encode(text), reloaded.Encode(text));
            }
            finally
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
        }
    }
}
