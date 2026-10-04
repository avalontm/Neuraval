using System;
using System.IO;

namespace Neuraval.Samples.DinoGame.Sources
{
    internal static class SampleDiagnostics
    {
        private static readonly object Gate = new object();

        private static readonly string LogDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Neuraval.Samples.DinoGame");

        private static readonly string LogFilePath = Path.Combine(LogDirectory, "dino-game.log");

        public static void Warn(string message, Exception exception)
        {
            string detail = exception == null
                ? message
                : $"{message} :: {exception.GetType().Name}: {exception.Message}";

            string line = $"{DateTime.UtcNow:o} WARN {detail}";

            System.Diagnostics.Debug.WriteLine(line);

            lock (Gate)
            {
                try
                {
                    Directory.CreateDirectory(LogDirectory);
                    File.AppendAllText(LogFilePath, line + Environment.NewLine);
                }
                catch (Exception writeFailure) when (writeFailure is IOException || writeFailure is UnauthorizedAccessException)
                {
                }
            }
        }
    }
}