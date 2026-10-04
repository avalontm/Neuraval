using System;
using System.Collections.Generic;
using System.IO;
using Neuraval.Core.Models;
using Neuraval.Core.Utils;

namespace Neuraval.Core.Serialization
{
    internal static class ModelStateBinaryConverter
    {
        internal static void WriteFloatArray(BinaryWriter writer, float[] array)
        {
            writer.Write(array.Length);
            if (array.Length == 0) return;

            var bytes = new byte[array.Length * sizeof(float)];
            Buffer.BlockCopy(array, 0, bytes, 0, bytes.Length);
            writer.Write(bytes);
        }

        internal static float[] ReadFloatArray(BinaryReader reader)
        {
            int length = reader.ReadInt32();
            if (length == 0) return Array.Empty<float>();

            var bytes = reader.ReadBytes(length * sizeof(float));
            var array = new float[length];
            Buffer.BlockCopy(bytes, 0, array, 0, bytes.Length);
            return array;
        }

        private static void WriteAdamMatrixOptimizerState(BinaryWriter writer, AdamMatrixOptimizerState state)
        {
            writer.Write(state.Rows);
            writer.Write(state.Cols);
            WriteFloatArray(writer, state.M);
            WriteFloatArray(writer, state.V);
            writer.Write(state.TimeStep);
        }

        private static AdamMatrixOptimizerState ReadAdamMatrixOptimizerState(BinaryReader reader)
        {
            return new AdamMatrixOptimizerState
            {
                Rows = reader.ReadInt32(),
                Cols = reader.ReadInt32(),
                M = ReadFloatArray(reader),
                V = ReadFloatArray(reader),
                TimeStep = reader.ReadInt32()
            };
        }

        private static void WriteOptionalAdamMatrixOptimizerState(BinaryWriter writer, AdamMatrixOptimizerState? state)
        {
            writer.Write(state != null);
            if (state != null)
            {
                WriteAdamMatrixOptimizerState(writer, state);
            }
        }

        private static AdamMatrixOptimizerState? ReadOptionalAdamMatrixOptimizerState(BinaryReader reader)
        {
            return reader.ReadBoolean() ? ReadAdamMatrixOptimizerState(reader) : null;
        }

        private static void WriteAdamVectorOptimizerState(BinaryWriter writer, AdamVectorOptimizerState state)
        {
            writer.Write(state.Length);
            WriteFloatArray(writer, state.M);
            WriteFloatArray(writer, state.V);
            writer.Write(state.TimeStep);
        }

        private static AdamVectorOptimizerState ReadAdamVectorOptimizerState(BinaryReader reader)
        {
            return new AdamVectorOptimizerState
            {
                Length = reader.ReadInt32(),
                M = ReadFloatArray(reader),
                V = ReadFloatArray(reader),
                TimeStep = reader.ReadInt32()
            };
        }

        private static void WriteOptionalAdamVectorOptimizerState(BinaryWriter writer, AdamVectorOptimizerState? state)
        {
            writer.Write(state != null);
            if (state != null)
            {
                WriteAdamVectorOptimizerState(writer, state);
            }
        }

        private static AdamVectorOptimizerState? ReadOptionalAdamVectorOptimizerState(BinaryReader reader)
        {
            return reader.ReadBoolean() ? ReadAdamVectorOptimizerState(reader) : null;
        }

        private static void WriteEmbeddingLayerState(BinaryWriter writer, EmbeddingLayerState state)
        {
            writer.Write(state.VocabSize);
            writer.Write(state.EmbeddingDim);
            WriteFloatArray(writer, state.Embeddings);
            WriteOptionalAdamMatrixOptimizerState(writer, state.OptimizerState);
        }

        private static EmbeddingLayerState ReadEmbeddingLayerState(BinaryReader reader)
        {
            return new EmbeddingLayerState
            {
                VocabSize = reader.ReadInt32(),
                EmbeddingDim = reader.ReadInt32(),
                Embeddings = ReadFloatArray(reader),
                OptimizerState = ReadOptionalAdamMatrixOptimizerState(reader)
            };
        }

        internal static void WriteLayerNormalizationState(BinaryWriter writer, LayerNormalizationState state)
        {
            writer.Write(state.NormalizedShape);
            writer.Write(state.Epsilon);
            WriteFloatArray(writer, state.Gamma);
            WriteFloatArray(writer, state.Beta);
            WriteOptionalAdamVectorOptimizerState(writer, state.GammaOptimizerState);
            WriteOptionalAdamVectorOptimizerState(writer, state.BetaOptimizerState);
        }

        internal static LayerNormalizationState ReadLayerNormalizationState(BinaryReader reader)
        {
            return new LayerNormalizationState
            {
                NormalizedShape = reader.ReadInt32(),
                Epsilon = reader.ReadSingle(),
                Gamma = ReadFloatArray(reader),
                Beta = ReadFloatArray(reader),
                GammaOptimizerState = ReadOptionalAdamVectorOptimizerState(reader),
                BetaOptimizerState = ReadOptionalAdamVectorOptimizerState(reader)
            };
        }

        internal static void WriteLoraProjectionState(BinaryWriter writer, LoraProjectionState state)
        {
            writer.Write(state.InDim);
            writer.Write(state.OutDim);
            writer.Write(state.Rank);
            writer.Write(state.Alpha);
            WriteFloatArray(writer, state.MatrixA);
            WriteFloatArray(writer, state.MatrixB);
            WriteOptionalAdamMatrixOptimizerState(writer, state.OptimizerAState);
            WriteOptionalAdamMatrixOptimizerState(writer, state.OptimizerBState);
        }

        internal static LoraProjectionState ReadLoraProjectionState(BinaryReader reader)
        {
            return new LoraProjectionState
            {
                InDim = reader.ReadInt32(),
                OutDim = reader.ReadInt32(),
                Rank = reader.ReadInt32(),
                Alpha = reader.ReadSingle(),
                MatrixA = ReadFloatArray(reader),
                MatrixB = ReadFloatArray(reader),
                OptimizerAState = ReadOptionalAdamMatrixOptimizerState(reader),
                OptimizerBState = ReadOptionalAdamMatrixOptimizerState(reader)
            };
        }

        internal static void WriteLoraAttentionState(BinaryWriter writer, LoraAttentionState state)
        {
            WriteLoraProjectionState(writer, state.Query);
            WriteLoraProjectionState(writer, state.Key);
            WriteLoraProjectionState(writer, state.Value);
            WriteLoraProjectionState(writer, state.Output);
            writer.Write(state.FreezeBaseWeights);
        }

        internal static LoraAttentionState ReadLoraAttentionState(BinaryReader reader)
        {
            return new LoraAttentionState
            {
                Query = ReadLoraProjectionState(reader),
                Key = ReadLoraProjectionState(reader),
                Value = ReadLoraProjectionState(reader),
                Output = ReadLoraProjectionState(reader),
                FreezeBaseWeights = reader.ReadBoolean()
            };
        }

        private static void WriteMultiHeadAttentionState(BinaryWriter writer, MultiHeadAttentionState state)
        {
            writer.Write(state.EmbeddingDim);
            writer.Write(state.NumHeads);
            WriteFloatArray(writer, state.QueryWeights);
            WriteFloatArray(writer, state.KeyWeights);
            WriteFloatArray(writer, state.ValueWeights);
            WriteFloatArray(writer, state.OutputWeights);
            WriteOptionalAdamMatrixOptimizerState(writer, state.QueryOptimizerState);
            WriteOptionalAdamMatrixOptimizerState(writer, state.KeyOptimizerState);
            WriteOptionalAdamMatrixOptimizerState(writer, state.ValueOptimizerState);
            WriteOptionalAdamMatrixOptimizerState(writer, state.OutputOptimizerState);

            writer.Write(state.LoraState != null);
            if (state.LoraState != null)
            {
                WriteLoraAttentionState(writer, state.LoraState);
            }
        }

        private static MultiHeadAttentionState ReadMultiHeadAttentionState(BinaryReader reader, ushort formatVersion)
        {
            var state = new MultiHeadAttentionState
            {
                EmbeddingDim = reader.ReadInt32(),
                NumHeads = reader.ReadInt32(),
                QueryWeights = ReadFloatArray(reader),
                KeyWeights = ReadFloatArray(reader),
                ValueWeights = ReadFloatArray(reader),
                OutputWeights = ReadFloatArray(reader),
                QueryOptimizerState = ReadOptionalAdamMatrixOptimizerState(reader),
                KeyOptimizerState = ReadOptionalAdamMatrixOptimizerState(reader),
                ValueOptimizerState = ReadOptionalAdamMatrixOptimizerState(reader),
                OutputOptimizerState = ReadOptionalAdamMatrixOptimizerState(reader)
            };

            if (formatVersion >= 2)
            {
                bool hasLora = reader.ReadBoolean();
                state.LoraState = hasLora ? ReadLoraAttentionState(reader) : null;
            }

            return state;
        }

        private static void WriteFeedForwardNetworkState(BinaryWriter writer, FeedForwardNetworkState state)
        {
            writer.Write(state.EmbeddingDim);
            writer.Write(state.HiddenDim);
            WriteFloatArray(writer, state.Weights1);
            WriteFloatArray(writer, state.Bias1);
            WriteFloatArray(writer, state.Weights2);
            WriteFloatArray(writer, state.Bias2);
            WriteOptionalAdamMatrixOptimizerState(writer, state.Weights1OptimizerState);
            WriteOptionalAdamVectorOptimizerState(writer, state.Bias1OptimizerState);
            WriteOptionalAdamMatrixOptimizerState(writer, state.Weights2OptimizerState);
            WriteOptionalAdamVectorOptimizerState(writer, state.Bias2OptimizerState);
        }

        private static FeedForwardNetworkState ReadFeedForwardNetworkState(BinaryReader reader)
        {
            return new FeedForwardNetworkState
            {
                EmbeddingDim = reader.ReadInt32(),
                HiddenDim = reader.ReadInt32(),
                Weights1 = ReadFloatArray(reader),
                Bias1 = ReadFloatArray(reader),
                Weights2 = ReadFloatArray(reader),
                Bias2 = ReadFloatArray(reader),
                Weights1OptimizerState = ReadOptionalAdamMatrixOptimizerState(reader),
                Bias1OptimizerState = ReadOptionalAdamVectorOptimizerState(reader),
                Weights2OptimizerState = ReadOptionalAdamMatrixOptimizerState(reader),
                Bias2OptimizerState = ReadOptionalAdamVectorOptimizerState(reader)
            };
        }

        private static void WriteTransformerBlockState(BinaryWriter writer, TransformerBlockState state)
        {
            writer.Write(state.EmbeddingDim);
            writer.Write(state.NumHeads);
            writer.Write(state.FeedforwardDim);
            writer.Write(state.Dropout);
            WriteMultiHeadAttentionState(writer, state.AttentionState);
            WriteFeedForwardNetworkState(writer, state.FeedforwardState);
            WriteLayerNormalizationState(writer, state.Norm1State);
            WriteLayerNormalizationState(writer, state.Norm2State);
        }

        private static TransformerBlockState ReadTransformerBlockState(BinaryReader reader, ushort formatVersion)
        {
            return new TransformerBlockState
            {
                EmbeddingDim = reader.ReadInt32(),
                NumHeads = reader.ReadInt32(),
                FeedforwardDim = reader.ReadInt32(),
                Dropout = reader.ReadSingle(),
                AttentionState = ReadMultiHeadAttentionState(reader, formatVersion),
                FeedforwardState = ReadFeedForwardNetworkState(reader),
                Norm1State = ReadLayerNormalizationState(reader),
                Norm2State = ReadLayerNormalizationState(reader)
            };
        }

        public static void WriteTransformerModelState(BinaryWriter writer, TransformerModelState state)
        {
            writer.Write(state.VocabSize);
            writer.Write(state.EmbeddingDim);
            writer.Write(state.NumLayers);
            writer.Write(state.NumHeads);
            writer.Write(state.FeedforwardDim);
            writer.Write(state.MaxSequenceLength);
            writer.Write(state.Dropout);

            WriteEmbeddingLayerState(writer, state.EmbeddingState);

            writer.Write(state.BlockStates.Count);
            foreach (var blockState in state.BlockStates)
            {
                WriteTransformerBlockState(writer, blockState);
            }

            WriteLayerNormalizationState(writer, state.FinalNormState);
            WriteFloatArray(writer, state.OutputBias);
            WriteOptionalAdamVectorOptimizerState(writer, state.OutputBiasOptimizerState);
        }

        public static TransformerModelState ReadTransformerModelState(BinaryReader reader, ushort formatVersion)
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
                EmbeddingState = ReadEmbeddingLayerState(reader)
            };

            int blockCount = reader.ReadInt32();
            var blockStates = new List<TransformerBlockState>(blockCount);
            for (int i = 0; i < blockCount; i++)
            {
                blockStates.Add(ReadTransformerBlockState(reader, formatVersion));
            }
            state.BlockStates = blockStates;

            state.FinalNormState = ReadLayerNormalizationState(reader);
            state.OutputBias = ReadFloatArray(reader);
            state.OutputBiasOptimizerState = ReadOptionalAdamVectorOptimizerState(reader);

            return state;
        }
    }
}
