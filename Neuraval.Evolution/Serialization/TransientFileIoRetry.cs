using System;
using System.IO;
using System.Threading;

namespace Neuraval.Evolution.Serialization
{
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
