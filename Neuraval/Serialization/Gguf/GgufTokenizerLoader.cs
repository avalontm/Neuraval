using System;
using System.Collections.Generic;
using Neuraval.Core.Tokenizers;

namespace Neuraval.Core.Serialization.Gguf
{
    public static class GgufTokenizerLoader
    {
        private const string Gpt2TokenizerModel = "gpt2";
        private const string SentencePieceTokenizerModel = "llama";

        public static IChatTokenizer Load(string filePath)
        {
            return Load(GgufReader.Read(filePath));
        }

        public static IChatTokenizer Load(GgufFile file)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));

            if (!file.Metadata.TryGetString("tokenizer.ggml.model", out var tokenizerModel) || string.IsNullOrEmpty(tokenizerModel))
                throw new InvalidOperationException("El GGUF no declara 'tokenizer.ggml.model'");

            if (string.Equals(tokenizerModel, Gpt2TokenizerModel, StringComparison.Ordinal))
                return LoadGpt2Tokenizer(file);

            if (string.Equals(tokenizerModel, SentencePieceTokenizerModel, StringComparison.Ordinal))
                return LoadSentencePieceTokenizer(file);

            throw new NotSupportedException(
                $"El tokenizer GGML '{tokenizerModel}' no está soportado por GgufTokenizerLoader; solo se soportan tokenizers byte-level BPE ('{Gpt2TokenizerModel}') y SentencePiece BPE ('{SentencePieceTokenizerModel}')");
        }

        private static ModernBpeTokenizer LoadGpt2Tokenizer(GgufFile file)
        {
            var tokenToId = ReadVocabulary(file);
            var merges = ReadMerges(file);
            var tokenTypes = ReadTokenTypes(file, tokenToId.Count);

            int bosToken = RequireMetadataTokenId(file, "tokenizer.ggml.bos_token_id");
            int eosToken = RequireMetadataTokenId(file, "tokenizer.ggml.eos_token_id");
            int unkToken = ResolveTokenId(file, "tokenizer.ggml.unknown_token_id", tokenToId, SpecialTokens.Unk, eosToken);
            int padToken = ResolveTokenId(file, "tokenizer.ggml.padding_token_id", tokenToId, SpecialTokens.Pad, eosToken);
            int sepToken = ResolveTokenId(file, "tokenizer.ggml.separator_token_id", tokenToId, SpecialTokens.Sep, eosToken);
            int imStartToken = ResolveTokenFromVocabulary(tokenToId, SpecialTokens.ImStart, bosToken);
            int imEndToken = ResolveTokenFromVocabulary(tokenToId, SpecialTokens.ImEnd, eosToken);

            var state = new ModernBpeTokenizerState
            {
                Vocabulary = new VocabularyState { TokenToId = tokenToId },
                Merges = merges,
                TokenTypes = tokenTypes,
                PadToken = padToken,
                UnknownToken = unkToken,
                StartToken = bosToken,
                EndToken = eosToken,
                SepToken = sepToken,
                ImStartToken = imStartToken,
                ImEndToken = imEndToken
            };

            return ModernBpeTokenizer.LoadState(state);
        }

        private static SentencePieceBpeTokenizer LoadSentencePieceTokenizer(GgufFile file)
        {
            var tokenToId = ReadVocabulary(file);
            var scores = ReadScores(file, tokenToId.Count);
            var tokenTypes = ReadTokenTypes(file, tokenToId.Count);

            int bosToken = RequireMetadataTokenId(file, "tokenizer.ggml.bos_token_id");
            int eosToken = RequireMetadataTokenId(file, "tokenizer.ggml.eos_token_id");
            int unkToken = ResolveTokenId(file, "tokenizer.ggml.unknown_token_id", tokenToId, SpecialTokens.Unk, eosToken);
            int padToken = ResolveTokenId(file, "tokenizer.ggml.padding_token_id", tokenToId, SpecialTokens.Pad, eosToken);
            int sepToken = ResolveTokenId(file, "tokenizer.ggml.separator_token_id", tokenToId, SpecialTokens.Sep, eosToken);
            int imStartToken = ResolveTokenFromVocabulary(tokenToId, SpecialTokens.ImStart, bosToken);
            int imEndToken = ResolveTokenFromVocabulary(tokenToId, SpecialTokens.ImEnd, eosToken);

            bool addDummyPrefix = !file.Metadata.TryGetValue("tokenizer.ggml.add_space_prefix", out var addSpaceEntry)
                || !addSpaceEntry.TryGetUInt64(out var addSpaceRaw)
                || addSpaceRaw != 0;

            var state = new SentencePieceBpeTokenizerState
            {
                Vocabulary = new VocabularyState { TokenToId = tokenToId },
                Scores = scores,
                TokenTypes = tokenTypes,
                AddDummyPrefix = addDummyPrefix,
                PadToken = padToken,
                UnknownToken = unkToken,
                StartToken = bosToken,
                EndToken = eosToken,
                SepToken = sepToken,
                ImStartToken = imStartToken,
                ImEndToken = imEndToken
            };

            return SentencePieceBpeTokenizer.LoadState(state);
        }

        private static Dictionary<string, int> ReadVocabulary(GgufFile file)
        {
            var tokens = ReadStringArray(file, "tokenizer.ggml.tokens");
            if (tokens.Count == 0)
                throw new InvalidOperationException("El GGUF no contiene 'tokenizer.ggml.tokens'");

            var tokenToId = new Dictionary<string, int>(tokens.Count);
            for (int i = 0; i < tokens.Count; i++)
                tokenToId[tokens[i]] = i;

            return tokenToId;
        }

        private static Dictionary<int, float> ReadScores(GgufFile file, int tokenCount)
        {
            var result = new Dictionary<int, float>(tokenCount);

            if (!file.Metadata.TryGetValue("tokenizer.ggml.scores", out var entry) || !entry.TryGetArray(out var array))
                throw new InvalidOperationException(
                    "El GGUF no contiene 'tokenizer.ggml.scores', requerido para reconstruir un tokenizer SentencePiece");

            for (int i = 0; i < array.Count && i < tokenCount; i++)
            {
                if (array[i].TryGetDouble(out var raw))
                    result[i] = (float)raw;
            }

            return result;
        }

        private static Dictionary<int, byte> ReadTokenTypes(GgufFile file, int tokenCount)
        {
            var result = new Dictionary<int, byte>(tokenCount);

            if (!file.Metadata.TryGetValue("tokenizer.ggml.token_type", out var entry) || !entry.TryGetArray(out var array))
                return result;

            for (int i = 0; i < array.Count && i < tokenCount; i++)
            {
                if (array[i].TryGetUInt64(out var raw))
                    result[i] = (byte)raw;
            }

            return result;
        }

        private static List<BpeMergeRule> ReadMerges(GgufFile file)
        {
            var rawMerges = ReadStringArray(file, "tokenizer.ggml.merges");
            var merges = new List<BpeMergeRule>(rawMerges.Count);

            foreach (var rawMerge in rawMerges)
            {
                var parts = rawMerge.Split(' ', 2);
                if (parts.Length != 2 || parts[0].Length == 0 || parts[1].Length == 0)
                    throw new InvalidOperationException($"Entrada de merge inválida en 'tokenizer.ggml.merges': '{rawMerge}'");

                merges.Add(new BpeMergeRule { First = parts[0], Second = parts[1] });
            }

            return merges;
        }

        private static List<string> ReadStringArray(GgufFile file, string key)
        {
            if (!file.Metadata.TryGetValue(key, out var entry) || !entry.TryGetArray(out var array))
                return new List<string>();

            var result = new List<string>(array.Count);

            foreach (var element in array)
            {
                if (!element.TryGetString(out var value))
                    throw new InvalidOperationException($"El array '{key}' del GGUF contiene un elemento que no es string");

                result.Add(value);
            }

            return result;
        }

        private static bool TryGetMetadataTokenId(GgufFile file, string key, out int id)
        {
            if (file.Metadata.TryGetValue(key, out var entry) && entry.TryGetUInt64(out var raw))
            {
                id = (int)raw;
                return true;
            }

            id = -1;
            return false;
        }

        private static int RequireMetadataTokenId(GgufFile file, string key)
        {
            if (TryGetMetadataTokenId(file, key, out var id))
                return id;

            throw new InvalidOperationException($"El GGUF no declara '{key}', requerido para reconstruir el tokenizer");
        }

        private static int ResolveTokenId(GgufFile file, string metadataKey, Dictionary<string, int> tokenToId, string literalFallback, int structuralFallback)
        {
            if (TryGetMetadataTokenId(file, metadataKey, out var id))
                return id;

            return ResolveTokenFromVocabulary(tokenToId, literalFallback, structuralFallback);
        }

        private static int ResolveTokenFromVocabulary(Dictionary<string, int> tokenToId, string literalToken, int structuralFallback)
        {
            return tokenToId.TryGetValue(literalToken, out var id) ? id : structuralFallback;
        }
    }
}
