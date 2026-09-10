namespace Neuraval.Core.Quantization
{
    /// <summary>
    /// Activa/desactiva el uso del kernel de matmul INT8 en CPU durante
    /// inferencia (<c>MultiHeadAttention.ForwardInference</c> /
    /// <c>ForwardIncremental</c>), análogo a <c>MixedPrecisionSettings</c>
    /// (FP16 en GPU, Fase 4.3) pero para el caso CPU de la Fase 5.4.
    ///
    /// A diferencia de <c>MixedPrecisionSettings</c> (que viene habilitado
    /// por defecto), este arranca <b>deshabilitado</b>: el kernel INT8 actual
    /// es una implementación escalar correcta pero todavía no vectorizada, así
    /// que no hay garantía de que sea más rápido que el backend CPU con SIMD
    /// ya existente (Fase 3) — solo garantiza pesos 4x más chicos en memoria.
    /// Conviene medir con <c>--int8-benchmark</c> antes de activarlo por
    /// defecto en un entorno productivo.
    /// </summary>
    public static class Int8InferenceSettings
    {
        private static bool _enableInt8Cpu;

        public static bool EnableInt8Cpu => _enableInt8Cpu;

        public static void Enable()
        {
            _enableInt8Cpu = true;
        }

        public static void Disable()
        {
            _enableInt8Cpu = false;
        }
    }
}
