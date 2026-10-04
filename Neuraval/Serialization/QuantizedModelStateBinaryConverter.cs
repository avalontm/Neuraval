using System;
using System.Collections.Generic;
using System.IO;
using Neuraval.Core.Models;
using Neuraval.Core.Quantization;

namespace Neuraval.Core.Serialization
{
    internal static class QuantizedModelStateBinaryConverter
    {
        private static void WriteQuantizedArray(BinaryWriter writer, float[] flatRowMajor, int rows, int cols)
        {
            var quantized = Int8Quantizer.QuantizeRowSymmetric(flatRowMajor, rows, cols);

            writer.Write(rows);
            writer.Write(cols);

            writer.Write(quantized.Data.Length);
            if (quantized.Data.Length > 0)
            {
                var bytes = new byte[quantized.Data.Length];
                Buffer.BlockCopy(quantized.Data, 0, bytes, 0, bytes.Length);
                writer.Write(bytes);
            }

            ModelStateBinaryConverter.WriteFloatArray(writer, quantized.RowScales);
        }

        private static float[] ReadQuantizedArray(BinaryReader reader)
        {
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();

            int dataLength = reader.ReadInt32();
            var data = new sbyte[dataLength];
            if (dataLength > 0)
            {
                var bytes = reader.ReadBytes(dataLength);
                Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
            }

            var rowScales = ModelStateBinaryConverter.ReadFloatArray(reader);

            var quantized = new QuantizedMatrix(rows, cols, data, rowScales);
            return Int8Quantizer.Dequantize(quantized);
        }

        private static void WriteEmbeddingLayerStateQuantized(BinaryWriter writer, EmbeddingLayerState state)
        {
            writer.Write(state.VocabSize);
            writer.Write(state.EmbeddingDim);
            WriteQuantizedArray(writer, state.Embeddings, state.VocabSize, state.EmbeddingDim);
        }

        private static EmbeddingLayerState ReadEmbeddingLayerStateQuantized(BinaryReader reader)
        {
            var state = new EmbeddingLayerState
            {
                VocabSize = reader.ReadInt32(),
                EmbeddingDim = reader.ReadInt32()
            };
            state.Embeddings = ReadQuantizedArray(reader);
            state.OptimizerState = null;
            return state;
        }

        private static void WriteMultiHeadAttentionStateQuantized(BinaryWriter writer, MultiHeadAttentionState state)
        {
            writer.Write(state.EmbeddingDim);
            writer.Write(state.NumHeads);
            int dim = state.EmbeddingDim;
            WriteQuantizedArray(writer, state.QueryWeights, dim, dim);
            WriteQuantizedArray(writer, state.KeyWeights, dim, dim);
            WriteQuantizedArray(writer, state.ValueWeights, dim, dim);
            WriteQuantizedArray(writer, state.OutputWeights, dim, dim);
        }

        private static MultiHeadAttentionState ReadMultiHeadAttentionStateQuantized(BinaryReader reader)
        {
            var state = new MultiHeadAttentionState
            {
                EmbeddingDim = reader.ReadInt32(),
                NumHeads = reader.ReadInt32()
            };
            state.QueryWeights = ReadQuantizedArray(reader);
            state.KeyWeights = ReadQuantizedArray(reader);
            state.ValueWeights = ReadQuantizedArray(reader);
            state.OutputWeights = ReadQuantizedArray(reader);
            state.QueryOptimizerState = null;
            state.KeyOptimizerState = null;
            state.ValueOptimizerState = null;
            state.OutputOptimizerState = null;
            return state;
        }

        private static void WriteFeedForwardNetworkStateQuantized(BinaryWriter writer, FeedForwardNetworkState state)
        {
            writer.Write(state.EmbeddingDim);
            writer.Write(state.HiddenDim);
            WriteQuantizedArray(writer, state.Weights1, state.EmbeddingDim, state.HiddenDim);
            ModelStateBinaryConverter.WriteFloatArray(writer, state.Bias1);
            WriteQuantizedArray(writer, state.Weights2, state.HiddenDim, state.EmbeddingDim);
            ModelStateBinaryConverter.WriteFloatArray(writer, state.Bias2);
        }

        private static FeedForwardNetworkState ReadFeedForwardNetworkStateQuantized(BinaryReader reader)
        {
            var state = new FeedForwardNetworkState
            {
                EmbeddingDim = reader.ReadInt32(),
                HiddenDim = reader.ReadInt32()
            };
            state.Weights1 = ReadQuantizedArray(reader);
            state.Bias1 = ModelStateBinaryConverter.ReadFloatArray(reader);
            state.Weights2 = ReadQuantizedArray(reader);
            state.Bias2 = ModelStateBinaryConverter.ReadFloatArray(reader);
            state.Weights1OptimizerState = null;
            state.Bias1OptimizerState = null;
            state.Weights2OptimizerState = null;
            state.Bias2OptimizerState = null;
            return state;
        }

        private static void WriteTransformerBlockStateQuantized(BinaryWriter writer, TransformerBlockState state)
        {
            writer.Write(state.EmbeddingDim);
            writer.Write(state.NumHeads);
            writer.Write(state.FeedforwardDim);
            writer.Write(state.Dropout);
            WriteMultiHeadAttentionStateQuantized(writer, state.AttentionState);
            WriteFeedForwardNetworkStateQuantized(writer, state.FeedforwardState);
            ModelStateBinaryConverter.WriteLayerNormalizationState(writer, state.Norm1State);
            ModelStateBinaryConverter.WriteLayerNormalizationState(writer, state.Norm2State);
        }

        private static TransformerBlockState ReadTransformerBlockStateQuantized(BinaryReader reader)
        {
            return new TransformerBlockState
            {
                EmbeddingDim = reader.ReadInt32(),
                NumHeads = reader.ReadInt32(),
                FeedforwardDim = reader.ReadInt32(),
                Dropout = reader.ReadSingle(),
                AttentionState = ReadMultiHeadAttentionStateQuantized(reader),
                FeedforwardState = ReadFeedForwardNetworkStateQuantized(reader),
                Norm1State = ModelStateBinaryConverter.ReadLayerNormalizationState(reader),
                Norm2State = ModelStateBinaryConverter.ReadLayerNormalizationState(reader)
            };
        }

        public static void WriteTransformerModelStateQuantized(BinaryWriter writer, TransformerModelState state)
        {
            writer.Write(state.VocabSize);
            writer.Write(state.EmbeddingDim);
            writer.Write(state.NumLayers);
            writer.Write(state.NumHeads);
            writer.Write(state.FeedforwardDim);
            writer.Write(state.MaxSequenceLength);
            writer.Write(state.Dropout);

            WriteEmbeddingLayerStateQuantized(writer, state.EmbeddingState);

            writer.Write(state.BlockStates.Count);
            foreach (var blockState in state.BlockStates)
            {
                WriteTransformerBlockStateQuantized(writer, blockState);
            }

            ModelStateBinaryConverter.WriteLayerNormalizationState(writer, state.FinalNormState);
            ModelStateBinaryConverter.WriteFloatArray(writer, state.OutputBias);
        }

        public static TransformerModelState ReadTransformerModelStateQuantized(BinaryReader reader)
        {
            var state = new TransformerModelState
            {
                VocabSize = reader.ReadInt32(),
                EmbeddingDim = reader.ReadInt32(),
                NumLayers = reader.ReadInt32(),
                NumHeads = reader.ReadInt32(),
                FeedforwardDim = reader.ReadInt32(),
                MaxSequenceLength = reader.ReadInt32(),
                Dropout = reader.ReadSingle(),
                EmbeddingState = ReadEmbeddingLayerStateQuantized(reader)
            };

            int blockCount = reader.ReadInt32();
            var blockStates = new List<TransformerBlockState>(blockCount);
            for (int i = 0; i < blockCount; i++)
            {
                blockStates.Add(ReadTransformerBlockStateQuantized(reader));
            }
            state.BlockStates = blockStates;

            state.FinalNormState = ModelStateBinaryConverter.ReadLayerNormalizationState(reader);
            state.OutputBias = ModelStateBinaryConverter.ReadFloatArray(reader);
            state.OutputBiasOptimizerState = null;

            return state;
        }
    }
}
