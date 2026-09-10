using System.Collections.Generic;
using System.IO;
using Neuraval.Core.Models;

namespace Neuraval.Core.Serialization
{
    /// <summary>
    /// Convierte un <see cref="TransformerModelLoraState"/> (los adaptadores
    /// LoRA de todos los bloques de un modelo, sin los pesos base) hacia/desde
    /// el layout binario posicional descrito en <see cref="LoraBinaryFormat"/>.
    ///
    /// Reutiliza los helpers de bajo nivel de <see cref="ModelStateBinaryConverter"/>
    /// (arreglos de floats, estado de Adam, <see cref="LoraProjectionState"/> y
    /// <see cref="LoraAttentionState"/>) para no duplicar ese código: el único
    /// layout nuevo acá es el envoltorio a nivel de modelo completo.
    /// </summary>
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
