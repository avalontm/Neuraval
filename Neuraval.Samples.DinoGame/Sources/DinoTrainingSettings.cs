using System;
using System.IO;
using System.Text.Json;

namespace Neuraval.Samples.DinoGame.Sources
{
    public class DinoTrainingSettings
    {
        public int PopulationSize { get; set; } = 1000;

        public int EliteCount { get; set; } = 20;

        public float MutationRate { get; set; } = 0.12f;

        public float RandomInjectionFraction { get; set; } = 0.2f;

        public int MaxEliteGenomesToSave { get; set; } = 60;

        public float SpeedStart { get; set; } = 12f;

        public float SpeedRampPerSecond { get; set; } = 0.00000025f;

        public int EnemySpawnIntervalFrames { get; set; } = 60;

        public float BirdSpawnProbabilityPercent { get; set; } = 20f;

        public int EnemySpawnIntervalJitterFrames { get; set; } = 45;

        public int EnemySpawnXJitter { get; set; } = 150;

        public int DinoStartX { get; set; } = 200;

        public int DinoStartXJitter { get; set; } = 80;

        private static readonly string SettingsDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Neuraval.Samples.DinoGame");

        private static readonly string SettingsFilePath = Path.Combine(SettingsDirectory, "dino-training-settings.json");

        public static DinoTrainingSettings Load()
        {
            if (!File.Exists(SettingsFilePath))
            {
                var defaults = new DinoTrainingSettings();
                Save(defaults);
                return defaults;
            }

            string json;

            try
            {
                json = File.ReadAllText(SettingsFilePath);
            }
            catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException)
            {
                SampleDiagnostics.Warn("dino-training-settings.json could not be read", exception);
                return new DinoTrainingSettings();
            }

            try
            {
                var settings = JsonSerializer.Deserialize<DinoTrainingSettings>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                return settings ?? new DinoTrainingSettings();
            }
            catch (JsonException exception)
            {
                SampleDiagnostics.Warn("dino-training-settings.json is not valid JSON; defaults are being used", exception);
                return new DinoTrainingSettings();
            }
        }

        public static void Save(DinoTrainingSettings settings)
        {
            try
            {
                Directory.CreateDirectory(SettingsDirectory);

                string json = JsonSerializer.Serialize(settings, new JsonSerializerOptions
                {
                    WriteIndented = true
                });

                File.WriteAllText(SettingsFilePath, json);
            }
            catch (Exception exception) when (exception is IOException || exception is UnauthorizedAccessException)
            {
                SampleDiagnostics.Warn("dino-training-settings.json could not be written", exception);
            }
        }

        public static string FilePath => SettingsFilePath;
    }
}
