using System.Collections.Generic;
using System.IO;
using Neuraval.Core.Models;

namespace Neuraval.Core.Serialization
{
    internal static class LoraStateBinaryConverter
    {
        public static void WriteTransformerModelLoraState(BinaryWriter writer, TransformerModelLoraState state)
        {
            writer.Write(state.NumLayers);
            writer.Write(state.EmbeddingDim);
            writer.Write(state.FreezeNonLoraWeights);

            writer.Write(state.BlockStates.Count);
            foreach (var blockState in state.BlockStates)
            {
                ModelStateBinaryConverter.WriteLoraAttentionState(writer, blockState);
            }
        }

        public static TransformerModelLoraState ReadTransformerModelLoraState(BinaryReader reader)
        {
            var state = new TransformerModelLoraState
            {
                NumLayers = reader.ReadInt32(),
                EmbeddingDim = reader.ReadInt32(),
                FreezeNonLoraWeights = reader.ReadBoolean()
            };

            int blockCount = reader.ReadInt32();
            var blockStates = new List<LoraAttentionState>(blockCount);
            for (int i = 0; i < blockCount; i++)
            {
                blockStates.Add(ModelStateBinaryConverter.ReadLoraAttentionState(reader));
            }
            state.BlockStates = blockStates;

            return state;
        }
    }
}
