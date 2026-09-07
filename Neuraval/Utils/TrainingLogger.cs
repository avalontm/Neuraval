using System;
using System.IO;

namespace Neuraval.Core.Utils
{
    public static class TrainingLogger
    {
        private const string Header = "epoch,training_loss,validation_loss,training_perplexity,learning_rate,gradient_norm,timestamp_utc";

        public static void LogEpoch(
            string filePath,
            int epoch,
            float trainingLoss,
            float validationLoss,
            float learningRate,
            float gradientNorm)
        {
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            bool writeHeader = !File.Exists(filePath);

            using var writer = new StreamWriter(filePath, append: true);

            if (writeHeader)
            {
                writer.WriteLine(Header);
            }

            float trainingPerplexity = MathF.Exp(trainingLoss);

            writer.WriteLine(string.Join(",",
                epoch,
                trainingLoss.ToString("F6"),
                validationLoss.ToString("F6"),
                trainingPerplexity.ToString("F6"),
                learningRate.ToString("F6"),
                gradientNorm.ToString("F6"),
                DateTime.UtcNow.ToString("o")));
        }
    }
}
