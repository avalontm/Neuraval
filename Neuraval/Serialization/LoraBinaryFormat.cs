using System;

namespace Neuraval.Core.Serialization
{
    /// <summary>
    /// Formato binario propietario de Neuraval para exportar/importar
    /// únicamente adaptadores LoRA (Fase 5.5), por separado del modelo base
    /// al que pertenecen.
    ///
    /// Extensión oficial: <c>.navlora</c> ("Neural Avalon LoRA").
    ///
    /// Comparte estructura con <see cref="ModelBinaryFormat"/> (mismo estilo
    /// de encabezado JSON + Body binario + checksum), pero con su propia
    /// firma mágica y su propio versionado de Body, independientes: un
    /// archivo <c>.navlora</c> nunca contiene pesos base, solo las matrices
    /// A/B (y su estado de Adam) de cada proyección Q/K/V/O de cada bloque.
    ///
    /// <code>
    ///   Offset   Tamaño   Campo
    ///   ------   ------   -----------------------------------------------------
    ///   0        4        Magic          -> bytes ASCII "NAVL"
    ///   4        2        FormatVersion  -> ushort, versión del layout del Body
    ///   6        1        Flags          -> byte (bit 0 = Body comprimido con GZip)
    ///   7        1        Reserved       -> byte, siempre 0 (uso futuro)
    ///   8        4        HeaderLength   -> int32, longitud en bytes del HeaderJson
    ///   12       N        HeaderJson     -> UTF-8 JSON (ver <see cref="LoraBinaryHeader"/>)
    ///   12+N     4        BodyLength     -> int32, longitud en bytes del Body
    ///   16+N     M        Body           -> adaptadores LoRA, crudo o comprimido
    ///   16+N+M   32       Checksum       -> SHA-256 del Body tal cual está en disco
    /// </code>
    /// </summary>
    public static class LoraBinaryFormat
    {
        /// <summary>Extensión de archivo oficial del formato.</summary>
        public const string FileExtension = ".navlora";

        /// <summary>Firma mágica que identifica el archivo como adaptadores LoRA de Neuraval.</summary>
        public static readonly byte[] MagicBytes = { (byte)'N', (byte)'A', (byte)'V', (byte)'L' };

        /// <summary>
        /// Versión actual del layout binario del Body. Se incrementa
        /// únicamente cuando cambia el orden/tipo de los campos escritos por
        /// <see cref="LoraStateBinaryConverter"/>.
        /// </summary>
        public const ushort CurrentFormatVersion = 1;

        /// <summary>Tamaño en bytes de un checksum SHA-256.</summary>
        public const int ChecksumLength = 32;

        [Flags]
        public enum LoraFlags : byte
        {
            None = 0,
            GZipCompressed = 1 << 0
        }
    }
}
