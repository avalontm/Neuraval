using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using Neuraval.Core.Models;
using Neuraval.Core.Tokenizers;

namespace Neuraval.Core.Serialization;

/// <summary>Compact, versioned native container for inference-ready modern decoder models.</summary>
public static class ModernDecoderBinarySerializer
{
    public const string FileExtension = ModelBinaryFormat.FileExtension;
    private const uint Magic = 0x4D44564E; // NV D M, little endian
    // Version 2 stores FP16, version 3 Q8, and version 4 keeps small norm/bias vectors in FP16.
    private const int Version = 4;
    private const int IoChunkSize = 1024 * 1024;
    private const int QuantizationBlockSize = 32;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        Converters = { new JsonStringEnumConverter() }
    };

    public static bool HasModernSignature(string path)
    {
        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            if (stream.Length < sizeof(uint)) return false;
            Span<byte> magic = stackalloc byte[sizeof(uint)];
            return stream.Read(magic) == magic.Length
                && System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(magic) == Magic;
        }
        catch (IOException)
        {
            return false;
        }
    }

    public static void Save(
        string path,
        ModernDecoderModelState state,
        IChatTokenizer tokenizer,
        ChatTemplateDefinition chatTemplate)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(chatTemplate);
        state.Config.Validate();
        ValidateStateShape(state);

        string tokenizerType;
        object tokenizerState;
        if (tokenizer is ModernBpeTokenizer modernBpe)
        {
            tokenizerType = "modern-bpe";
            tokenizerState = modernBpe.SaveState();
        }
        else if (tokenizer is SentencePieceBpeTokenizer sentencePiece)
        {
            tokenizerType = "sentencepiece-bpe";
            tokenizerState = sentencePiece.SaveState();
        }
        else
        {
            throw new NotSupportedException($"Tokenizer type '{tokenizer.GetType().Name}' no soportado en {FileExtension}.");
        }

        byte[] tokenizerJson = JsonSerializer.SerializeToUtf8Bytes(tokenizerState, tokenizerState.GetType(), JsonOptions);
        using var tokenizerDocument = JsonDocument.Parse(tokenizerJson);
        var metadata = new ModelMetadata
        {
            Config = state.Config,
            ChatTemplate = chatTemplate,
            TokenizerType = tokenizerType,
            TokenizerState = tokenizerDocument.RootElement.Clone()
        };
        byte[] metadataJson = JsonSerializer.SerializeToUtf8Bytes(metadata, JsonOptions);

        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        string temporaryPath = fullPath + ".tmp";
        try
        {
            using (var stream = new FileStream(temporaryPath, FileMode.Create, FileAccess.Write, FileShare.None, IoChunkSize))
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            using (var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
            {
                writer.Write(Magic);
                writer.Write(Version);
                writer.Write(metadataJson.Length);
                writer.Write(metadataJson);
                WriteStateBody(writer, state, hash);
                writer.Write(hash.GetHashAndReset());
                writer.Flush();
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporaryPath, fullPath, overwrite: true);
        }
        catch
        {
            if (File.Exists(temporaryPath)) File.Delete(temporaryPath);
            throw;
        }
    }

    public static ModernDecoderFile Load(string path, bool inferenceOnly = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, IoChunkSize, FileOptions.SequentialScan);
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        if (reader.ReadUInt32() != Magic)
            throw new InvalidDataException($"El archivo no tiene la firma Neuraval {FileExtension} esperada.");
        int version = reader.ReadInt32();
        if (version is < 1 or > Version)
            throw new NotSupportedException($"Versión {version} de {FileExtension} no soportada.");

        int metadataLength = reader.ReadInt32();
        if (metadataLength <= 0 || metadataLength > 64 * 1024 * 1024 || metadataLength > stream.Length - stream.Position)
            throw new InvalidDataException("La cabecera del modelo Neuraval tiene un tamaño inválido.");
        var metadata = JsonSerializer.Deserialize<ModelMetadata>(reader.ReadBytes(metadataLength), JsonOptions)
            ?? throw new InvalidDataException("No se pudo leer la configuración del modelo Neuraval.");
        metadata.Config.Validate();

        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var state = ReadStateBody(reader, metadata.Config, hash, version);
        byte[] expectedHash = reader.ReadBytes(32);
        byte[] actualHash = hash.GetHashAndReset();
        if (expectedHash.Length != 32 || !CryptographicOperations.FixedTimeEquals(expectedHash, actualHash))
            throw new InvalidDataException("La suma de comprobación del modelo Neuraval no coincide.");
        if (stream.Position != stream.Length)
            throw new InvalidDataException("El archivo contiene datos sobrantes después del modelo.");

        IChatTokenizer tokenizer = metadata.TokenizerType switch
        {
            "modern-bpe" => ModernBpeTokenizer.LoadState(
                metadata.TokenizerState.Deserialize<ModernBpeTokenizerState>(JsonOptions)
                ?? throw new InvalidDataException("El tokenizer BPE del modelo está vacío.")),
            "sentencepiece-bpe" => SentencePieceBpeTokenizer.LoadState(
                metadata.TokenizerState.Deserialize<SentencePieceBpeTokenizerState>(JsonOptions)
                ?? throw new InvalidDataException("El tokenizer SentencePiece del modelo está vacío.")),
            _ => throw new InvalidDataException($"Tipo de tokenizer Neuraval desconocido: '{metadata.TokenizerType}'.")
        };

        return new ModernDecoderFile(ModernDecoderModel.LoadState(state, inferenceOnly), tokenizer, metadata.ChatTemplate);
    }

    private static void WriteStateBody(BinaryWriter writer, ModernDecoderModelState state, IncrementalHash hash)
    {
        WriteFloatArray(writer, state.EmbeddingState.Embeddings, hash);
        WriteHalfArray(writer, state.FinalNormState.Weight, hash);

        foreach (var block in state.BlockStates)
        {
            WriteHalfArray(writer, block.Norm1State.Weight, hash);
            WriteHalfArray(writer, block.Norm2State.Weight, hash);
            WriteMatrix(writer, block.AttentionState.Wq, hash);
            WriteMatrix(writer, block.AttentionState.Wk, hash);
            WriteMatrix(writer, block.AttentionState.Wv, hash);
            WriteMatrix(writer, block.AttentionState.Wo, hash);
            WriteHalfArray(writer, block.AttentionState.Bq, hash);
            WriteHalfArray(writer, block.AttentionState.Bk, hash);
            WriteHalfArray(writer, block.AttentionState.Bv, hash);
            WriteFloatArray(writer, block.FeedforwardState.WeightsGate, hash);
            WriteFloatArray(writer, block.FeedforwardState.WeightsUp, hash);
            WriteFloatArray(writer, block.FeedforwardState.WeightsDown, hash);
        }

        if (!state.Config.TieWordEmbeddings)
            WriteFloatArray(writer, state.OutputWeights ?? throw new InvalidDataException("Faltan los pesos de salida."), hash);
    }

    private static ModernDecoderModelState ReadStateBody(BinaryReader reader, TransformerConfig config, IncrementalHash hash, int version)
    {
        int hidden = config.HiddenSize;
        int kvDim = config.NumKeyValueHeads * config.HeadDim;
        int vocabHidden = checked(config.VocabSize * hidden);
        var state = new ModernDecoderModelState
        {
            Config = config,
            EmbeddingState = new EmbeddingLayerState
            {
                VocabSize = config.VocabSize,
                EmbeddingDim = hidden,
                Embeddings = ReadFloatArray(reader, vocabHidden, hash, version)
            },
            FinalNormState = new RMSNormState
            {
                NormalizedShape = hidden,
                Epsilon = config.RmsNormEps,
                Weight = ReadFloatArray(reader, hidden, hash, version, quantized: false)
            },
            BlockStates = new List<ModernDecoderBlockState>(config.NumHiddenLayers)
        };

        for (int layer = 0; layer < config.NumHiddenLayers; layer++)
        {
            var norm1 = new RMSNormState { NormalizedShape = hidden, Epsilon = config.RmsNormEps, Weight = ReadFloatArray(reader, hidden, hash, version, quantized: false) };
            var norm2 = new RMSNormState { NormalizedShape = hidden, Epsilon = config.RmsNormEps, Weight = ReadFloatArray(reader, hidden, hash, version, quantized: false) };
            var attention = new GQAAttentionState
            {
                HiddenSize = hidden,
                NumAttentionHeads = config.NumAttentionHeads,
                NumKeyValueHeads = config.NumKeyValueHeads,
                Wq = ReadMatrix(reader, hidden, hidden, hash, version),
                Wk = ReadMatrix(reader, hidden, kvDim, hash, version),
                Wv = ReadMatrix(reader, hidden, kvDim, hash, version),
                Wo = ReadMatrix(reader, hidden, hidden, hash, version),
                Bq = ReadFloatArray(reader, hidden, hash, version, quantized: false),
                Bk = ReadFloatArray(reader, kvDim, hash, version, quantized: false),
                Bv = ReadFloatArray(reader, kvDim, hash, version, quantized: false)
            };
            var feedforward = new SwiGLUFeedForwardState
            {
                EmbeddingDim = hidden,
                HiddenDim = config.IntermediateSize,
                WeightsGate = ReadFloatArray(reader, checked(hidden * config.IntermediateSize), hash, version),
                WeightsUp = ReadFloatArray(reader, checked(hidden * config.IntermediateSize), hash, version),
                WeightsDown = ReadFloatArray(reader, checked(hidden * config.IntermediateSize), hash, version)
            };
            state.BlockStates.Add(new ModernDecoderBlockState
            {
                Config = config.Clone(),
                Seed = config.Seed + layer + 1,
                Norm1State = norm1,
                Norm2State = norm2,
                AttentionState = attention,
                FeedforwardState = feedforward
            });
        }

        if (!config.TieWordEmbeddings)
            state.OutputWeights = ReadFloatArray(reader, vocabHidden, hash, version);
        return state;
    }

    private static void WriteFloatArray(BinaryWriter writer, float[] values, IncrementalHash hash)
    {
        writer.Write(values.Length);
        WriteQuantizedValues(writer, values, hash);
    }

    private static void WriteHalfArray(BinaryWriter writer, float[] values, IncrementalHash hash)
    {
        writer.Write(values.Length);
        WriteHalfValues(writer, values, hash);
    }

    // Symmetric Q8 in groups of 32: FP16 scale + 32 signed bytes (close to GGUF Q8_0).
    private static void WriteQuantizedValues(BinaryWriter writer, ReadOnlySpan<float> values, IncrementalHash hash)
    {
        Span<byte> quantized = stackalloc byte[QuantizationBlockSize];
        Span<byte> scaleBytes = stackalloc byte[sizeof(ushort)];
        for (int offset = 0; offset < values.Length; offset += QuantizationBlockSize)
        {
            int count = Math.Min(QuantizationBlockSize, values.Length - offset);
            float maxAbs = 0f;
            for (int i = 0; i < count; i++)
                maxAbs = MathF.Max(maxAbs, MathF.Abs(values[offset + i]));

            float scale = maxAbs / 127f;
            ushort halfScale = BitConverter.HalfToUInt16Bits((Half)scale);
            System.Buffers.Binary.BinaryPrimitives.WriteUInt16LittleEndian(scaleBytes, halfScale);
            writer.BaseStream.Write(scaleBytes);
            hash.AppendData(scaleBytes);

            for (int i = 0; i < count; i++)
            {
                int q = scale == 0f ? 0 : (int)MathF.Round(values[offset + i] / scale);
                quantized[i] = unchecked((byte)(sbyte)Math.Clamp(q, -127, 127));
            }
            var block = quantized.Slice(0, count);
            writer.BaseStream.Write(block);
            hash.AppendData(block);
        }
    }

    private static void WriteHalfValues(BinaryWriter writer, ReadOnlySpan<float> values, IncrementalHash hash)
    {
        byte[] buffer = new byte[Math.Min(IoChunkSize, Math.Max(sizeof(ushort), values.Length * sizeof(ushort)))];
        int valuesPerChunk = buffer.Length / sizeof(ushort);
        for (int offset = 0; offset < values.Length; offset += valuesPerChunk)
        {
            int count = Math.Min(valuesPerChunk, values.Length - offset);
            Span<byte> bytes = buffer.AsSpan(0, count * sizeof(ushort));
            for (int i = 0; i < count; i++)
                System.Buffers.Binary.BinaryPrimitives.WriteUInt16LittleEndian(bytes.Slice(i * 2, 2), BitConverter.HalfToUInt16Bits((Half)values[offset + i]));
            writer.BaseStream.Write(bytes);
            hash.AppendData(bytes);
        }
    }

    private static float[] ReadFloatArray(BinaryReader reader, int expectedLength, IncrementalHash hash, int version, bool quantized = true)
    {
        int length = reader.ReadInt32();
        if (length != expectedLength)
            throw new InvalidDataException($"Longitud de tensor inválida: se esperaban {expectedLength} valores y hay {length}.");
        bool storeAsHalf = version == 2 || (version >= 4 && !quantized);
        int byteLength = version switch
        {
            1 => checked(length * sizeof(float)),
            _ when storeAsHalf => checked(length * sizeof(ushort)),
            _ when version == 3 || version == 4 => GetQuantizedByteLength(length),
            _ => throw new NotSupportedException($"Versión {version} de pesos no soportada.")
        };
        EnsurePayloadAvailable(reader, byteLength);
        var values = new float[length];
        if (version == 1)
            ReadPayload(reader, System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()), hash);
        else if (storeAsHalf)
            ReadHalfValues(reader, values, hash);
        else
            ReadQuantizedValues(reader, values, hash);
        return values;
    }

    private static void ReadQuantizedValues(BinaryReader reader, Span<float> values, IncrementalHash hash)
    {
        Span<byte> scaleBytes = stackalloc byte[sizeof(ushort)];
        Span<byte> quantized = stackalloc byte[QuantizationBlockSize];
        for (int offset = 0; offset < values.Length; offset += QuantizationBlockSize)
        {
            int count = Math.Min(QuantizationBlockSize, values.Length - offset);
            ReadHashed(reader, scaleBytes, hash);
            float scale = (float)BitConverter.UInt16BitsToHalf(
                System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(scaleBytes));
            ReadHashed(reader, quantized.Slice(0, count), hash);
            for (int i = 0; i < count; i++)
                values[offset + i] = unchecked((sbyte)quantized[i]) * scale;
        }
    }

    private static int GetQuantizedByteLength(int valueCount)
    {
        int blockCount = checked((valueCount + QuantizationBlockSize - 1) / QuantizationBlockSize);
        return checked(valueCount + blockCount * sizeof(ushort));
    }

    private static void ReadHalfValues(BinaryReader reader, Span<float> values, IncrementalHash hash)
    {
        byte[] buffer = new byte[Math.Min(IoChunkSize, Math.Max(sizeof(ushort), values.Length * sizeof(ushort)))];
        int valuesPerChunk = buffer.Length / sizeof(ushort);
        for (int offset = 0; offset < values.Length; offset += valuesPerChunk)
        {
            int count = Math.Min(valuesPerChunk, values.Length - offset);
            Span<byte> bytes = buffer.AsSpan(0, count * sizeof(ushort));
            ReadHashed(reader, bytes, hash);
            for (int i = 0; i < count; i++)
                values[offset + i] = (float)BitConverter.UInt16BitsToHalf(System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(i * 2, 2)));
        }
    }

    private static void WriteMatrix(BinaryWriter writer, float[,] matrix, IncrementalHash hash)
    {
        int rows = matrix.GetLength(0), columns = matrix.GetLength(1);
        writer.Write(rows);
        writer.Write(columns);
        int elementCount = checked(rows * columns);
        int valuesPerChunk = IoChunkSize / sizeof(float);
        var values = new float[valuesPerChunk];
        for (int offset = 0; offset < elementCount; offset += values.Length)
        {
            int count = Math.Min(values.Length, elementCount - offset);
            Buffer.BlockCopy(matrix, offset * sizeof(float), values, 0, count * sizeof(float));
            WriteQuantizedValues(writer, values.AsSpan(0, count), hash);
        }
    }

    private static float[,] ReadMatrix(BinaryReader reader, int expectedRows, int expectedColumns, IncrementalHash hash, int version)
    {
        int rows = reader.ReadInt32(), columns = reader.ReadInt32();
        if (rows != expectedRows || columns != expectedColumns)
            throw new InvalidDataException($"Forma de matriz inválida: se esperaba [{expectedRows},{expectedColumns}] y hay [{rows},{columns}].");
        int elementCount = checked(rows * columns);
        int byteLength = version switch
        {
            1 => checked(elementCount * sizeof(float)),
            2 => checked(elementCount * sizeof(ushort)),
            _ => GetQuantizedByteLength(elementCount)
        };
        EnsurePayloadAvailable(reader, byteLength);
        var matrix = new float[rows, columns];
        if (version == 1)
        {
            int legacyByteLength = checked(elementCount * sizeof(float));
            var buffer = new byte[Math.Min(IoChunkSize, legacyByteLength)];
            for (int offset = 0; offset < legacyByteLength; offset += buffer.Length)
            {
                int count = Math.Min(buffer.Length, legacyByteLength - offset);
                ReadHashed(reader, buffer.AsSpan(0, count), hash);
                Buffer.BlockCopy(buffer, 0, matrix, offset, count);
            }
        }
        else if (version == 2)
        {
            int valuesPerChunk = IoChunkSize / sizeof(float);
            var values = new float[valuesPerChunk];
            for (int offset = 0; offset < elementCount; offset += values.Length)
            {
                int count = Math.Min(values.Length, elementCount - offset);
                ReadHalfValues(reader, values.AsSpan(0, count), hash);
                Buffer.BlockCopy(values, 0, matrix, offset * sizeof(float), count * sizeof(float));
            }
        }
        else
        {
            int valuesPerChunk = IoChunkSize / sizeof(float);
            var values = new float[valuesPerChunk];
            for (int offset = 0; offset < elementCount; offset += values.Length)
            {
                int count = Math.Min(values.Length, elementCount - offset);
                ReadQuantizedValues(reader, values.AsSpan(0, count), hash);
                Buffer.BlockCopy(values, 0, matrix, offset * sizeof(float), count * sizeof(float));
            }
        }
        return matrix;
    }

    private static void WritePayload(BinaryWriter writer, ReadOnlySpan<byte> payload, IncrementalHash hash)
    {
        for (int offset = 0; offset < payload.Length; offset += IoChunkSize)
        {
            var chunk = payload.Slice(offset, Math.Min(IoChunkSize, payload.Length - offset));
            writer.BaseStream.Write(chunk);
            hash.AppendData(chunk);
        }
    }

    private static void ReadPayload(BinaryReader reader, Span<byte> payload, IncrementalHash hash)
    {
        for (int offset = 0; offset < payload.Length; offset += IoChunkSize)
            ReadHashed(reader, payload.Slice(offset, Math.Min(IoChunkSize, payload.Length - offset)), hash);
    }

    private static void ReadHashed(BinaryReader reader, Span<byte> destination, IncrementalHash hash)
    {
        reader.BaseStream.ReadExactly(destination);
        hash.AppendData(destination);
    }

    private static void EnsurePayloadAvailable(BinaryReader reader, int byteLength)
    {
        const int checksumLength = 32;
        if (byteLength < 0 || reader.BaseStream.Length - reader.BaseStream.Position - checksumLength < byteLength)
            throw new InvalidDataException("El archivo termina antes de completar un tensor.");
    }

    private static void ValidateStateShape(ModernDecoderModelState state)
    {
        if (state.BlockStates.Count != state.Config.NumHiddenLayers)
            throw new InvalidDataException("El número de bloques no coincide con la configuración.");
        int hidden = state.Config.HiddenSize;
        int kvDim = state.Config.NumKeyValueHeads * state.Config.HeadDim;
        if (state.EmbeddingState.Embeddings.Length != checked(state.Config.VocabSize * hidden)
            || state.FinalNormState.Weight.Length != hidden)
            throw new InvalidDataException("Los pesos de embedding o normalización no coinciden con la configuración.");
        foreach (var block in state.BlockStates)
        {
            if (block.Norm1State.Weight.Length != hidden || block.Norm2State.Weight.Length != hidden
                || block.AttentionState.Wq.GetLength(0) != hidden || block.AttentionState.Wq.GetLength(1) != hidden
                || block.AttentionState.Wk.GetLength(0) != hidden || block.AttentionState.Wk.GetLength(1) != kvDim
                || block.AttentionState.Wv.GetLength(0) != hidden || block.AttentionState.Wv.GetLength(1) != kvDim
                || block.AttentionState.Wo.GetLength(0) != hidden || block.AttentionState.Wo.GetLength(1) != hidden
                || block.AttentionState.Bq.Length != hidden || block.AttentionState.Bk.Length != kvDim
                || block.AttentionState.Bv.Length != kvDim
                || block.FeedforwardState.WeightsGate.Length != checked(hidden * state.Config.IntermediateSize)
                || block.FeedforwardState.WeightsUp.Length != checked(hidden * state.Config.IntermediateSize)
                || block.FeedforwardState.WeightsDown.Length != checked(hidden * state.Config.IntermediateSize))
                throw new InvalidDataException("Los pesos de una capa de atención no coinciden con la configuración.");
        }
        if (!state.Config.TieWordEmbeddings && state.OutputWeights?.Length != checked(state.Config.VocabSize * hidden))
            throw new InvalidDataException("Los pesos de salida no coinciden con la configuración.");
    }

    private sealed class ModelMetadata
    {
        public TransformerConfig Config { get; set; } = new();
        public ChatTemplateDefinition ChatTemplate { get; set; } = new();
        public string TokenizerType { get; set; } = string.Empty;
        public JsonElement TokenizerState { get; set; }
    }
}

public sealed record ModernDecoderFile(ModernDecoderModel Model, IChatTokenizer Tokenizer, ChatTemplateDefinition ChatTemplate);
