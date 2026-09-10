using System;
using System.Collections.Generic;
using System.IO;
using Neuraval.Core.Models;
using Neuraval.Core.Quantization;

namespace Neuraval.Core.Serialization
{
    /// <summary>
    /// Convierte un <see cref="TransformerModelState"/> hacia/desde un layout
    /// binario cuantizado en INT8 (Fase 5.4, post-entrenamiento).
    ///
    /// Diferencias respecto a <see cref="ModelStateBinaryConverter"/> (el
    /// formato de precisión completa):
    /// <list type="bullet">
    /// <item>Las matrices de pesos grandes (embeddings, proyecciones Q/K/V/O
    /// de atención, capas feed-forward) se guardan como INT8 + una escala
    /// por fila (<see cref="Int8Quantizer"/>), en vez de float32 crudo.</item>
    /// <item>Los vectores chicos (bias, gamma/beta de layer norm) se dejan en
    /// float32: su aporte al tamaño total es marginal y cuantizarlos
    /// arriesga más precisión de la que ahorra.</item>
    /// <item>No se guarda estado de optimizadores Adam: un modelo cuantizado
    /// es un artefacto de solo-inferencia, no pensado para reanudar
    /// entrenamiento. Si hace falta seguir entrenando, se debe partir del
    /// <c>.navm</c> sin cuantizar.</item>
    /// </list>
    ///
    /// Al cargar, las matrices se decuantizan de vuelta a float32 en memoria:
    /// el resto del pipeline (Forward, ForwardInference, ForwardIncremental)
    /// no necesita saber que el archivo en disco estaba cuantizado. Esta es
    /// la única sub-fase de 5.4 (5.4.1: reducir el tamaño en disco); los
    /// kernels de matmul que operen directamente sobre INT8 en memoria
    /// (para acelerar inferencia además de achicar el archivo) quedan para
    /// una sub-fase siguiente.
    /// </summary>
    internal static class QuantizedModelStateBinaryConverter
    {
        // ---------------------------------------------------------------
        // Helper: matriz grande cuantizada (row-major, rows x cols)
        // ---------------------------------------------------------------

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

        // ---------------------------------------------------------------
        // EmbeddingLayerState
        // ---------------------------------------------------------------

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

        // ---------------------------------------------------------------
        // MultiHeadAttentionState
        // ---------------------------------------------------------------

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

        // ---------------------------------------------------------------
        // FeedForwardNetworkState
        // ---------------------------------------------------------------

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

        // ---------------------------------------------------------------
        // TransformerBlockState
        // ---------------------------------------------------------------

        private static void WriteTransformerBlockStateQuantized(BinaryWriter writer, TransformerBlockState state)
        {
            writer.Write(state.EmbeddingDim);
            writer.Write(state.NumHeads);
            writer.Write(state.FeedforwardDim);
            writer.Write(state.Dropout);
            WriteMultiHeadAttentionStateQuantized(writer, state.AttentionState);
            WriteFeedForwardNetworkStateQuantized(writer, state.FeedforwardState);
            // Layer norm queda en precisión completa: son solo dos vectores
            // de tamaño EmbeddingDim por bloque, y son sensibles porque
            // reescalan la salida de cada sub-capa.
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

        // ---------------------------------------------------------------
        // TransformerModelState (punto de entrada)
        // ---------------------------------------------------------------

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

            // FinalNormState y OutputBias quedan en precisión completa:
            // ninguno de los dos pesa lo suficiente (a lo sumo VocabSize
            // floats) como para justificar cuantizarlos.
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
