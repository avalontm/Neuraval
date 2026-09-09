using System;
using System.IO;
using System.Threading;

namespace Neuraval.Evolution.Serialization
{
    // En Windows es muy comun que, apenas se crea o reemplaza un archivo,
    // otro proceso (Windows Defender u otro antivirus, OneDrive/Dropbox,
    // el indexador de busqueda) lo abra un instante para escanearlo o
    // indexarlo. Si justo en ese momento el codigo intenta otra operacion
    // sobre el mismo archivo (un File.Move o File.Replace, por ejemplo),
    // falla con IOException: "El proceso no puede acceder al archivo
    // porque esta siendo usado por otro proceso" — aunque el archivo este
    // perfectamente bien y el bloqueo dure apenas milisegundos.
    //
    // Este helper reintenta la operacion unas pocas veces con espera
    // creciente antes de darse por vencido, para absorber ese tipo de
    // bloqueo transitorio sin perder un guardado (checkpoint o modelo)
    // por una casualidad de timing.
    public static class TransientFileIoRetry
    {
        public static void Run(Action action, int maxAttempts = 5, int initialDelayMs = 150, Action<int, int, IOException>? onRetry = null)
        {
            var delayMs = initialDelayMs;
            for (var attempt = 1; attempt <= maxAttempts; attempt++)
            {
                try
                {
                    action();
                    return;
                }
                catch (IOException ex) when (attempt < maxAttempts)
                {
                    onRetry?.Invoke(attempt, maxAttempts - 1, ex);
                    Thread.Sleep(delayMs);
                    delayMs = Math.Min(delayMs * 2, 2000);
                }
            }
        }
    }
}
