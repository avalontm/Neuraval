using System;

namespace Neuraval.Core.Serialization
{
    /// <summary>
    /// Formato binario propietario de Neuraval para persistir los pesos
    /// entrenados de un <see cref="Models.TransformerModel"/> (y el estado de
    /// sus optimizadores Adam) en un único archivo compacto.
    ///
    /// Extensión oficial: <c>.navm</c> ("Neural Avalon Model").
    ///
    /// Layout del archivo en disco (little-endian; todo escrito con
    /// <see cref="System.IO.BinaryWriter"/>):
    ///
    /// <code>
    ///   Offset   Tamaño   Campo
    ///   ------   ------   -----------------------------------------------------
    ///   0        4        Magic          -> bytes ASCII "NAVM"
    ///   4        2        FormatVersion  -> ushort, versión del layout del Body
    ///   6        1        Flags          -> byte (bit 0 = Body comprimido con GZip)
    ///   7        1        Reserved       -> byte, siempre 0 (uso futuro)
    ///   8        4        HeaderLength   -> int32, longitud en bytes del HeaderJson
    ///   12       N        HeaderJson     -> UTF-8 JSON (ver <see cref="ModelBinaryHeader"/>)
    ///   12+N     4        BodyLength     -> int32, longitud en bytes del Body (tal
    ///                                       cual queda escrito en disco)
    ///   16+N     M        Body           -> pesos + estado de optimizadores,
    ///                                       crudo o comprimido con GZip según Flags
    ///   16+N+M   32       Checksum       -> SHA-256 del Body tal cual está en disco
    /// </code>
    ///
    /// Decisiones de diseño:
    /// <list type="bullet">
    /// <item>El encabezado va en JSON, sin comprimir: permite inspeccionar la
    /// arquitectura y metadatos del modelo (o escribir herramientas externas)
    /// sin tener que descomprimir ni parsear los pesos, y admite agregar campos
    /// nuevos sin romper compatibilidad binaria.</item>
    /// <item>El cuerpo (los pesos, que son la parte pesada) usa un layout binario
    /// posicional propio, versionado de forma independiente con
    /// <see cref="CurrentFormatVersion"/>, y los arreglos de floats se escriben
    /// como bloques de memoria crudos (no como texto), lo que lo hace mucho más
    /// compacto y rápido de leer/escribir que JSON a medida que el modelo crece.</item>
    /// <item>La compresión GZip es opcional (activada por defecto) y el checksum
    /// SHA-256 permite detectar archivos truncados o corruptos al cargar.</item>
    /// </list>
    /// </summary>
    public static class ModelBinaryFormat
    {
        /// <summary>Extensión de archivo oficial del formato.</summary>
        public const string FileExtension = ".navm";

        /// <summary>Firma mágica que identifica el archivo como un modelo Neuraval.</summary>
        public static readonly byte[] MagicBytes = { (byte)'N', (byte)'A', (byte)'V', (byte)'M' };

        /// <summary>
        /// Versión actual del layout binario del Body (pesos + optimizadores).
        /// Se incrementa únicamente cuando cambia el orden/tipo de los campos
        /// escritos por <see cref="ModelStateBinaryConverter"/>.
        ///
        /// Historial:
        /// <list type="bullet">
        /// <item>1: layout original (pesos + Adam), sin LoRA.</item>
        /// <item>2 (Fase 5.5): cada <c>MultiHeadAttentionState</c> agrega, al
        /// final, un flag opcional + el estado de sus adaptadores LoRA (si
        /// los tiene). Los archivos versión 1 se siguen leyendo igual: ese
        /// bloque extra simplemente no se lee y las capas quedan sin LoRA,
        /// como antes.</item>
        /// </list>
        /// </summary>
        public const ushort CurrentFormatVersion = 2;

        /// <summary>Tamaño en bytes de un checksum SHA-256.</summary>
        public const int ChecksumLength = 32;

        [Flags]
        public enum ModelFlags : byte
        {
            None = 0,
            GZipCompressed = 1 << 0,

            /// <summary>
            /// El Body fue escrito con <c>QuantizedModelStateBinaryConverter</c>:
            /// las matrices de pesos grandes (embeddings, proyecciones de
            /// atención, feed-forward) están en INT8 + escala por fila en vez
            /// de float32, y no incluye estado de optimizadores Adam (es un
            /// artefacto de solo-inferencia, ver Fase 5.4 del plan).
            /// </summary>
            Int8QuantizedWeights = 1 << 1
        }
    }
}
