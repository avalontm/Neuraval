using System;

namespace Neuraval.Core.Serialization
{
    /// <summary>
    /// Metadatos livianos de un modelo <c>.ncbm</c>, almacenados como JSON al
    /// inicio del archivo para que puedan inspeccionarse sin leer los pesos.
    /// No incluye ningún arreglo grande: eso vive únicamente en el Body binario.
    /// </summary>
    public class ModelBinaryHeader
    {
        /// <summary>Tamaño del vocabulario del tokenizer usado por el modelo.</summary>
        public int VocabSize { get; set; }

        /// <summary>Dimensión de los embeddings / del modelo (d_model).</summary>
        public int EmbeddingDim { get; set; }

        /// <summary>Cantidad de bloques Transformer apilados.</summary>
        public int NumLayers { get; set; }

        /// <summary>Cantidad de cabezas de atención por bloque.</summary>
        public int NumHeads { get; set; }

        /// <summary>Dimensión de la capa oculta feed-forward.</summary>
        public int FeedforwardDim { get; set; }

        /// <summary>Longitud máxima de secuencia soportada.</summary>
        public int MaxSequenceLength { get; set; }

        /// <summary>Cantidad de hilos configurada para el entrenamiento/inferencia.</summary>
        public int NumThreads { get; set; }

        /// <summary>Tipo de tokenizer asociado ("bpe" o "wordlevel").</summary>
        public string TokenizerType { get; set; } = "wordlevel";

        /// <summary>Indica si el modelo ya pasó por al menos un ciclo de entrenamiento.</summary>
        public bool IsTrained { get; set; }

        /// <summary>Fecha (UTC) en la que se guardó este archivo.</summary>
        public DateTime SavedAtUtc { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Versión del layout binario del Body con el que se guardó el archivo.
        /// Se completa automáticamente al guardar/cargar; no hace falta setearla a mano.
        /// </summary>
        public ushort FormatVersion { get; set; }

        /// <summary>Indica si el Body quedó comprimido con GZip en disco.</summary>
        public bool Compressed { get; set; }
    }
}
