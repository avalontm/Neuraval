using System;
using System.Collections.Generic;

namespace Neuraval.Core.Models
{
    /// <summary>
    /// Estado LoRA de un <see cref="TransformerModel"/> completo: un
    /// <see cref="LoraAttentionState"/> (Q/K/V/O) por cada bloque, en el mismo
    /// orden que <see cref="TransformerModel"/> los apila. Se usa tanto para
    /// persistir los adaptadores dentro del checkpoint normal (<c>.navm</c>,
    /// vía <see cref="MultiHeadAttentionState.LoraState"/> en cada bloque) como
    /// para exportarlos/importarlos por separado del modelo base con el
    /// formato standalone <c>.navlora</c>.
    /// </summary>
    public class TransformerModelLoraState
    {
        public int NumLayers { get; set; }
        public int EmbeddingDim { get; set; }

        /// <summary>
        /// Si true, el modelo del que salió este adaptador tenía la base
        /// (embedding, bloques salvo LoRA, norma final, bias de salida)
        /// congelada durante el fine-tuning.
        /// </summary>
        public bool FreezeNonLoraWeights { get; set; }

        public List<LoraAttentionState> BlockStates { get; set; } = new();
    }
}
