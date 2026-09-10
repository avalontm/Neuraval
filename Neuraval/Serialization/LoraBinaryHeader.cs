using System;

namespace Neuraval.Core.Serialization
{
    /// <summary>
    /// Metadatos livianos de un archivo <c>.navlora</c>, almacenados como JSON
    /// al inicio del archivo para poder inspeccionar el adaptador (a qué
    /// arquitectura de modelo le pertenece, su rank, etc.) sin decodificar las
    /// matrices A/B.
    /// </summary>
    public class LoraBinaryHeader
    {
        /// <summary>Dimensión de los embeddings / del modelo base (d_model) al que le pertenece este adaptador.</summary>
        public int EmbeddingDim { get; set; }

        /// <summary>Cantidad de bloques Transformer del modelo base.</summary>
        public int NumLayers { get; set; }

        /// <summary>Rank de los adaptadores LoRA (igual en las cuatro proyecciones de cada bloque).</summary>
        public int Rank { get; set; }

        /// <summary>Alpha de LoRA usado al entrenar este adaptador.</summary>
        public float Alpha { get; set; }

        /// <summary>Indica si la base del modelo estaba congelada mientras se entrenó este adaptador.</summary>
        public bool FreezeNonLoraWeights { get; set; }

        /// <summary>Fecha (UTC) en la que se guardó este archivo.</summary>
        public DateTime SavedAtUtc { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Versión del layout binario del Body con el que se guardó el
        /// archivo. Se completa automáticamente al guardar/cargar.
        /// </summary>
        public ushort FormatVersion { get; set; }

        /// <summary>Indica si el Body quedó comprimido con GZip en disco.</summary>
        public bool Compressed { get; set; }
    }
}
