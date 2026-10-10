using System;
using System.Collections.Generic;
using System.Linq;
using Neuraval.Abstractions;
using Neuraval.Core.Serialization.Gguf;
using Neuraval.Core.Tokenizers;
using Xunit;

namespace Neuraval.Tests
{
    public class GgufTokenizerLoaderTests
    {
        private static GgufMetadataValue Str(string value) => new(GgufValueType.String, value);
        private static GgufMetadataValue U32(uint value) => new(GgufValueType.UInt32, value);
        private static GgufMetadataValue F32(float value) => new(GgufValueType.Float32, value);
        private static GgufMetadataValue StrArray(params string[] values) =>
            new(GgufValueType.Array, values.Select(Str).ToList());
        private static GgufMetadataValue F32Array(params float[] values) =>
            new(GgufValueType.Array, values.Select(F32).ToList());
        private static GgufMetadataValue I32Array(params int[] values) =>
            new(GgufValueType.Array, values.Select(v => new GgufMetadataValue(GgufValueType.Int32, v)).ToList());

        private static readonly string[] BaseTokens =
        {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>", "a", "b", "c", "ab", "bc"
        };

        private static readonly string[] BaseMerges = { "a b", "ab c" };

        private static readonly string[] SpTokens =
        {
            "<unk>", "<s>", "</s>", "\u2581", "h", "i", "\u2581h", "\u2581hi"
        };

        private static readonly float[] SpScores = { 0f, 0f, 0f, -1f, -1f, -1f, -0.5f, -0.1f };
        private static readonly int[] SpTokenTypes = { 2, 3, 3, 1, 1, 1, 1, 1 };

        private static GgufFile BuildSentencePieceFile(
            string[]? tokens = null,
            float[]? scores = null,
            int[]? tokenTypes = null,
            uint bosTokenId = 1,
            uint eosTokenId = 2,
            uint unknownTokenId = 0)
        {
            tokens ??= SpTokens;
            scores ??= SpScores;
            tokenTypes ??= SpTokenTypes;

            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["tokenizer.ggml.model"] = Str("llama"),
                ["tokenizer.ggml.tokens"] = StrArray(tokens),
                ["tokenizer.ggml.scores"] = F32Array(scores),
                ["tokenizer.ggml.token_type"] = I32Array(tokenTypes),
                ["tokenizer.ggml.bos_token_id"] = U32(bosTokenId),
                ["tokenizer.ggml.eos_token_id"] = U32(eosTokenId),
                ["tokenizer.ggml.unknown_token_id"] = U32(unknownTokenId)
            };

            return new GgufFile(3, metadata, Array.Empty<GgufTensorEntry>());
        }

        private static GgufFile BuildFile(
            string tokenizerModel = "gpt2",
            string[]? tokens = null,
            string[]? merges = null,
            uint? bosTokenId = 0,
            uint? eosTokenId = 0,
            uint? unknownTokenId = null,
            uint? paddingTokenId = null,
            bool? addBosToken = null)
        {
            tokens ??= BaseTokens;
            merges ??= BaseMerges;

            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["tokenizer.ggml.model"] = Str(tokenizerModel),
                ["tokenizer.ggml.tokens"] = StrArray(tokens),
                ["tokenizer.ggml.merges"] = StrArray(merges)
            };

            if (bosTokenId.HasValue)
                metadata["tokenizer.ggml.bos_token_id"] = U32(bosTokenId.Value);

            if (eosTokenId.HasValue)
                metadata["tokenizer.ggml.eos_token_id"] = U32(eosTokenId.Value);

            if (unknownTokenId.HasValue)
                metadata["tokenizer.ggml.unknown_token_id"] = U32(unknownTokenId.Value);

            if (paddingTokenId.HasValue)
                metadata["tokenizer.ggml.padding_token_id"] = U32(paddingTokenId.Value);

            if (addBosToken.HasValue)
                metadata["tokenizer.ggml.add_bos_token"] = new GgufMetadataValue(GgufValueType.Bool, addBosToken.Value);

            return new GgufFile(3, metadata, Array.Empty<GgufTensorEntry>());
        }

        [Fact]
        public void Load_Gpt2Tokenizer_ReconstructsVocabularyAndMerges()
        {
            var file = BuildFile(bosTokenId: 0, eosTokenId: 0);

            var tokenizer = GgufTokenizerLoader.Load(file);

            Assert.Equal(BaseTokens.Length, tokenizer.VocabSize);
            Assert.Equal(0, tokenizer.StartToken);
            Assert.Equal(0, tokenizer.EndToken);
            Assert.Equal(3, tokenizer.GetTokenId("a"));
            Assert.Equal("bc", tokenizer.GetToken(7));
        }

        [Fact]
        public void Load_ImStartAndImEndPresentInVocabulary_ResolvedByLiteralLookup()
        {
            var file = BuildFile(bosTokenId: 0, eosTokenId: 0);

            var tokenizer = GgufTokenizerLoader.Load(file);

            Assert.Equal(1, tokenizer.ImStartToken);
            Assert.Equal(2, tokenizer.ImEndToken);
        }

        [Fact]
        public void Load_Gpt2Tokenizer_RespectsAddBosTokenFalseForChatPrompts()
        {
            var file = BuildFile(
                tokens: new[] { "<|endoftext|>", "<|im_start|>", "<|im_end|>", "a", "b", "c", "ab", "bc" },
                bosTokenId: 0,
                eosTokenId: 0,
                addBosToken: false);
            var tokenizer = GgufTokenizerLoader.Load(file);

            var ids = tokenizer.EncodeChat(new List<ChatMessage> { new(ChatRole.User, "hi") });

            Assert.False(tokenizer.AddBosToken);
            Assert.Equal(tokenizer.ImStartToken, ids[0]);
            Assert.NotEqual(tokenizer.StartToken, ids[0]);
        }

        [Fact]
        public void Load_UnknownAndPaddingAbsent_FallBackToEosTokenId()
        {
            var file = BuildFile(bosTokenId: 1, eosTokenId: 2, unknownTokenId: null, paddingTokenId: null);

            var tokenizer = GgufTokenizerLoader.Load(file);

            Assert.Equal(2, tokenizer.UnknownToken);
            Assert.Equal(2, tokenizer.PadToken);
        }

        [Fact]
        public void Load_UnknownTokenIdDeclared_UsesDeclaredValue()
        {
            var file = BuildFile(bosTokenId: 0, eosTokenId: 0, unknownTokenId: 5);

            var tokenizer = GgufTokenizerLoader.Load(file);

            Assert.Equal(5, tokenizer.UnknownToken);
        }

        [Fact]
        public void Load_UnsupportedTokenizerModel_ThrowsNotSupportedException()
        {
            var file = BuildFile(tokenizerModel: "bert");

            Assert.Throws<NotSupportedException>(() => GgufTokenizerLoader.Load(file));
        }

        [Fact]
        public void Load_SentencePieceModel_ReconstructsVocabularyAndScores()
        {
            var file = BuildSentencePieceFile();

            var tokenizer = GgufTokenizerLoader.Load(file);

            Assert.IsType<SentencePieceBpeTokenizer>(tokenizer);
            Assert.Equal(SpTokens.Length, tokenizer.VocabSize);
            Assert.Equal(1, tokenizer.StartToken);
            Assert.Equal(2, tokenizer.EndToken);
            Assert.Equal(0, tokenizer.UnknownToken);
        }

        [Fact]
        public void Load_SentencePieceModel_MergesHighestScoringPairsFirst()
        {
            var file = BuildSentencePieceFile();
            var tokenizer = GgufTokenizerLoader.Load(file);

            var ids = tokenizer.Encode("hi", addSpecialTokens: false);

            Assert.Equal(new[] { 7 }, ids);
        }

        [Fact]
        public void Load_SentencePieceModel_DecodeRestoresOriginalText()
        {
            var file = BuildSentencePieceFile();
            var tokenizer = GgufTokenizerLoader.Load(file);

            var ids = tokenizer.Encode("hi", addSpecialTokens: false);
            var text = tokenizer.Decode(ids, skipSpecialTokens: true);

            Assert.Equal("hi", text);
        }

        [Fact]
        public void Load_SentencePieceModel_EncodeChatStartsWithBosToken()
        {
            var file = BuildSentencePieceFile();
            var tokenizer = GgufTokenizerLoader.Load(file);

            var messages = new List<ChatMessage> { new ChatMessage(ChatRole.User, "hi") };
            var ids = tokenizer.EncodeChat(messages, addGenerationPrompt: false);

            Assert.Equal(tokenizer.StartToken, ids[0]);
        }

        [Fact]
        public void Load_SentencePieceModel_MissingScores_ThrowsInvalidOperationException()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["tokenizer.ggml.model"] = Str("llama"),
                ["tokenizer.ggml.tokens"] = StrArray(SpTokens),
                ["tokenizer.ggml.bos_token_id"] = U32(1),
                ["tokenizer.ggml.eos_token_id"] = U32(2)
            };
            var file = new GgufFile(3, metadata, Array.Empty<GgufTensorEntry>());

            Assert.Throws<InvalidOperationException>(() => GgufTokenizerLoader.Load(file));
        }

        [Fact]
        public void Load_MissingModelMetadata_ThrowsInvalidOperationException()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["tokenizer.ggml.tokens"] = StrArray(BaseTokens)
            };
            var file = new GgufFile(3, metadata, Array.Empty<GgufTensorEntry>());

            Assert.Throws<InvalidOperationException>(() => GgufTokenizerLoader.Load(file));
        }

        [Fact]
        public void Load_MissingTokensArray_ThrowsInvalidOperationException()
        {
            var metadata = new Dictionary<string, GgufMetadataValue>
            {
                ["tokenizer.ggml.model"] = Str("gpt2")
            };
            var file = new GgufFile(3, metadata, Array.Empty<GgufTensorEntry>());

            Assert.Throws<InvalidOperationException>(() => GgufTokenizerLoader.Load(file));
        }

        [Fact]
        public void Load_MissingBosTokenId_ThrowsInvalidOperationException()
        {
            var file = BuildFile(bosTokenId: null, eosTokenId: 0);

            Assert.Throws<InvalidOperationException>(() => GgufTokenizerLoader.Load(file));
        }

        [Fact]
        public void Load_MissingEosTokenId_ThrowsInvalidOperationException()
        {
            var file = BuildFile(bosTokenId: 0, eosTokenId: null);

            Assert.Throws<InvalidOperationException>(() => GgufTokenizerLoader.Load(file));
        }

        [Fact]
        public void Load_InvalidMergeEntry_ThrowsInvalidOperationException()
        {
            var file = BuildFile(merges: new[] { "onlyonepart" });

            Assert.Throws<InvalidOperationException>(() => GgufTokenizerLoader.Load(file));
        }
    }
}
