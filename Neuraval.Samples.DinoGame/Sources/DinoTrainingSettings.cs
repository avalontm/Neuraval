using System;
using System.IO;
using System.Text.Json;

namespace Neuraval.Samples.DinoGame.Sources
{
    /// <summary>
    /// Todos los parametros configurables del aprendizaje/evolucion del
    /// Dino, antes hardcodeados como "const" en <see cref="Neuraval.Samples.DinoGame.MainGame"/>
    /// y en varios otros archivos. Se cargan (y se crean con valores por
    /// defecto si no existen) desde un JSON en la misma carpeta donde ya se
    /// guarda el progreso de la evolucion (<see cref="DinoEvolutionStore"/>),
    /// para poder ajustarlos sin recompilar y que sobrevivan a
    /// reinstalaciones igual que el resto del progreso guardado.
    /// </summary>
    public class DinoTrainingSettings
    {
        // --- Poblacion / algoritmo genetico ---

        /// <summary>Cuantos dinosaurios (y por lo tanto cerebros) se crean por generacion.</summary>
        public int PopulationSize { get; set; } = 1000;

        /// <summary>
        /// Cuantos de los mejores dinosaurios de cada generacion pasan sin
        /// cambios (elitismo): garantiza que nunca se pierde lo que ya
        /// funcionaba bien de una generacion a la siguiente.
        /// </summary>
        public int EliteCount { get; set; } = 20;

        /// <summary>Probabilidad (0-1) de que cada peso/bias individual mute al reproducirse.</summary>
        public float MutationRate { get; set; } = 0.12f;

        /// <summary>Magnitud (desviacion estandar) de la mutacion gaussiana aplicada a cada peso mutado.</summary>
        public float MutationStrength { get; set; } = 0.35f;

        /// <summary>
        /// Fraccion (0-1) de cada generacion que se rellena con cerebros
        /// nuevos aleatorios en vez de descendencia de la elite, para no
        /// perder diversidad genetica generacion tras generacion. Se aplica
        /// tanto en la evolucion en vivo (<see cref="Neuraval.Evolution.ElitistMutationStrategy{TAgent}"/>)
        /// como al reconstruir la poblacion desde el guardado en disco
        /// (<see cref="DinoEvolutionStore.RebuildPopulation"/>).
        /// </summary>
        public float RandomInjectionFraction { get; set; } = 0.2f;

        /// <summary>
        /// Cuantos genomas de "elite" de la ultima generacion se guardan en
        /// disco (ademas del mejor historico) para poder repoblar con
        /// diversidad al reabrir el juego.
        /// </summary>
        public int MaxEliteGenomesToSave { get; set; } = 60;

        // --- Dificultad / entorno del juego ---

        /// <summary>Velocidad del juego al arrancar cada ronda.</summary>
        public float SpeedStart { get; set; } = 12f;

        /// <summary>
        /// Cuanto aumenta la velocidad por segundo transcurrido *de la
        /// ronda actual* (no del total de la sesion: ver el fix del bug de
        /// rampa de velocidad). Controla que tan rapido se vuelve dificil el
        /// juego a medida que pasa el tiempo.
        /// </summary>
        public float SpeedRampPerSecond { get; set; } = 0.00000025f;

        /// <summary>Cada cuantos fotogramas se evalua si aparece un nuevo enemigo.</summary>
        public int EnemySpawnIntervalFrames { get; set; } = 60;

        /// <summary>Probabilidad (0-100) de que el siguiente enemigo generado sea un pajaro en vez de un cactus.</summary>
        public float BirdSpawnProbabilityPercent { get; set; } = 20f;

        // --- Posicion inicial del Dino ---

        /// <summary>Posicion X base donde arranca cada dinosaurio al reiniciar.</summary>
        public int DinoStartX { get; set; } = 200;

        /// <summary>
        /// Rango de variacion aleatoria (+/-) sobre <see cref="DinoStartX"/>,
        /// para que no todos los dinosaurios arranquen exactamente alineados.
        /// </summary>
        public int DinoStartXJitter { get; set; } = 80;

        // --- Carga / guardado ---

        private static readonly string SettingsDirectory = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "Neuraval.Samples.DinoGame");

        private static readonly string SettingsFilePath = Path.Combine(SettingsDirectory, "dino-training-settings.json");

        /// <summary>
        /// Carga la configuracion desde disco. Si el archivo no existe (por
        /// ejemplo, primera vez que se ejecuta el juego), crea uno nuevo con
        /// los valores por defecto de arriba y lo guarda, para que el
        /// jugador pueda encontrarlo y editarlo facilmente.
        /// </summary>
        public static DinoTrainingSettings Load()
        {
            try
            {
                if (!File.Exists(SettingsFilePath))
                {
                    var defaults = new DinoTrainingSettings();
                    Save(defaults);
                    return defaults;
                }

                string json = File.ReadAllText(SettingsFilePath);
                var settings = JsonSerializer.Deserialize<DinoTrainingSettings>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                return settings ?? new DinoTrainingSettings();
            }
            catch
            {
                // Un JSON invalido o de una version incompatible no debe
                // impedir que el juego arranque: se usan los valores por
                // defecto, igual que si no existiera el archivo.
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
            catch
            {
                // Guardar la configuracion es "best effort", igual que
                // DinoEvolutionStore.Save: si falla, el juego sigue
                // funcionando con los valores ya cargados en memoria.
            }
        }

        /// <summary>Ruta del archivo, expuesta para poder mostrarla al usuario (p. ej. en un log de arranque).</summary>
        public static string FilePath => SettingsFilePath;
    }
}
