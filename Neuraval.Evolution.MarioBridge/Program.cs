using System.Globalization;
using System.Net;
using Neuraval.Evolution;
using Neuraval.Evolution.Neat;

namespace Neuraval.Evolution.MarioBridge
{
    internal static class Program
    {
        private const string Address = "127.0.0.1";
        private const int BasePort = 8766;
        private const int DefaultPopulationSize = 16;
        private const int MaxStepsPerEpisode = 3600;
        private const int ImitationEpochs = 20;
        private const int StagnationWatchGenerations = 10;
        private const string CheckpointPath = "checkpoints/mario_checkpoint.navm";
        private const string BestModelPath = "checkpoints/mario_best.navm";
        private const string CapturePath = "checkpoints/mario_dataset.navm";
        private const string PolicyPath = "checkpoints/mario_policy.navm";
        private const string FitnessHistoryPath = "checkpoints/fitness_history.csv";
        private const string GeneralizationReportPath = "checkpoints/generalization_report.csv";
        private const int DefaultEvaluationEpisodes = 10;

        private static readonly string[] DefaultLevels = { "Nivel 0 (DP1.state)" };

        private static void Main(string[] args)
        {
            if (args.Any(arg => arg.Equals("--help", StringComparison.OrdinalIgnoreCase) || arg.Equals("-h", StringComparison.OrdinalIgnoreCase)))
            {
                PrintUsage();
                return;
            }

            var freshRequested = args.Any(arg => arg.Equals("--fresh", StringComparison.OrdinalIgnoreCase));

            if (args.Any(arg => arg.Equals("--capture", StringComparison.OrdinalIgnoreCase)))
            {
                if (freshRequested)
                {
                    DeleteIfExists(CapturePath);
                    Console.WriteLine($"Dataset viejo borrado ({CapturePath}); la captura arranca limpia.");
                }

                RunCaptureMode();
                return;
            }

            if (args.Any(arg => arg.Equals("--imitate", StringComparison.OrdinalIgnoreCase)))
            {
                RunImitationMode();
                return;
            }

            if (args.Any(arg => arg.Equals("--play", StringComparison.OrdinalIgnoreCase)))
            {
                RunPlayMode(args);
                return;
            }

            if (args.Any(arg => arg.Equals("--evaluate", StringComparison.OrdinalIgnoreCase)))
            {
                RunEvaluateMode(args);
                return;
            }

            if (args.Any(arg => arg.Equals("--learn", StringComparison.OrdinalIgnoreCase)))
            {
                if (freshRequested)
                {
                    DeleteIfExists(CapturePath);
                    DeleteIfExists(PolicyPath);
                    Console.WriteLine("Dataset y politica anteriores borrados; el bucle arranca de cero.");
                }

                RunLearnMode(args);
                return;
            }

            var populationSize = TryGetInt(GetOption(args, "--population"), DefaultPopulationSize);
            var workers = Math.Max(1, TryGetInt(GetOption(args, "--workers"), 1));

            var connections = new SnesBridgeConnection[workers];
            for (var w = 0; w < workers; w++)
            {
                connections[w] = new SnesBridgeConnection(IPAddress.Parse(Address), BasePort + w);
            }

            try
            {
                RunTrainingModeCore(args, populationSize, workers, freshRequested, connections);
            }
            finally
            {
                foreach (var connection in connections)
                {
                    connection.Dispose();
                }
            }
        }

        private static void PrintUsage()
        {
            Console.WriteLine("MarioBridge - Neuraval.Evolution");
            Console.WriteLine();
            Console.WriteLine("Modos (si no pones ninguno, entrena por NEAT):");
            Console.WriteLine("  --capture    Graba tu juego (teclado) en checkpoints/mario_dataset.navm.");
            Console.WriteLine("  --imitate    Entrena por imitacion con el dataset -> checkpoints/mario_policy.navm.");
            Console.WriteLine("  --play       El modelo entrenado juega solo (turbo).");
            Console.WriteLine("  --learn      Bucle: captura -> entrena -> juega; Ctrl+C avanza de fase, Ctrl+C doble sale.");
            Console.WriteLine("  --evaluate   Corre N episodios por --level SIN entrenar ni modificar el modelo, y");
            Console.WriteLine("               reporta % completado / best X / muertes por nivel. Pensado para medir");
            Console.WriteLine("               generalizacion (Fase 5): pasale un --level en el que el modelo NO");
            Console.WriteLine("               entreno para ver si de verdad 'aprendio a jugar' o memorizo DP1.");
            Console.WriteLine("  --help, -h   Esta ayuda.");
            Console.WriteLine();
            Console.WriteLine("Opciones:");
            Console.WriteLine("  --level <nombre>    Nivel/savestate a rotar (repetible; default 'Nivel 0 (DP1.state)').");
            Console.WriteLine("  --model <ruta>      Modelo NEAT a usar con --evaluate (default " + BestModelPath + ").");
            Console.WriteLine("  --episodes <N>      Episodios por nivel con --evaluate (default " + DefaultEvaluationEpisodes + ").");
            Console.WriteLine("  --population <N>    Tamano de poblacion NEAT (default 16).");
            Console.WriteLine("  --seed-policy <ruta> Semilla NEAT desde una politica imitada.");
            Console.WriteLine("  --workers <N>       Instancias BizHawk en paralelo (default 1).");
            Console.WriteLine("  --fresh             Borra artefactos de corridas anteriores antes de empezar.");
            Console.WriteLine("  --curriculum        Modo entrenamiento (sin --learn/--play/--capture): en vez de rotar");
            Console.WriteLine("                      los --level por generacion, arranca en el primero y solo avanza");
            Console.WriteLine("                      al siguiente cuando el % de completions de una ventana de");
            Console.WriteLine("                      generaciones recientes supera el umbral. Se queda en el ultimo");
            Console.WriteLine("                      tramo configurado indefinidamente (no hay 'graduacion' automatica");
            Console.WriteLine("                      a otros niveles; agregalos vos a mano con mas --level).");
            Console.WriteLine("  --curriculum-window <N>     Generaciones en la ventana movil (default " + MarioCurriculum.DefaultWindowSize + ").");
            Console.WriteLine("  --curriculum-threshold <pct> % de completions en la ventana para avanzar de tramo (default " + MarioCurriculum.DefaultAdvanceThresholdPercent.ToString("F0", CultureInfo.InvariantCulture) + ").");
            Console.WriteLine();
            Console.WriteLine("Artifacts fijos:");
            Console.WriteLine($"  {CheckpointPath}  checkpoint de poblacion NEAT");
            Console.WriteLine($"  {BestModelPath}    mejor genoma exportado por NEAT");
            Console.WriteLine($"  {CapturePath}   dataset de imitacion (acumulativo)");
            Console.WriteLine($"  {PolicyPath}   politica imitada");
            Console.WriteLine($"  {FitnessHistoryPath} historial de fitness por generacion");
            Console.WriteLine($"  {GeneralizationReportPath} reporte de --evaluate (acumulativo)");
        }

        private static void RunCaptureMode()
        {
            using var connection = new SnesBridgeConnection(IPAddress.Parse(Address), BasePort);
            var stopRequested = false;

            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                stopRequested = true;
                Console.WriteLine("\nDeteniendo captura y guardando dataset...");
            };

            Console.WriteLine($"Esperando conexion de BizHawk en {Address}:{BasePort}...");
            try
            {
                connection.WaitForBizHawk(() => stopRequested);
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("Se cancelo la espera de conexion con BizHawk. No se grabo nada.");
                return;
            }

            Console.WriteLine("BizHawk conectado. Juega con el teclado: se graba tu juego junto con tus botones.");
            Console.WriteLine("Muertes, fin de nivel y el reset manual (tecla Insert) cierran el episodio.");
            Console.WriteLine("Ctrl+C para cerrar el dataset y salir.");

            using var recorder = new MarioDatasetRecorder(CapturePath, append: true);
            var sampler = new MarioFrameSampler();
            var summary = new MarioCaptureSummary();
            var state = connection.ReceiveState();
            int? previousX = null;
            int? previousCoins = null;
            int? previousPowerup = null;
            var frames = 0;
            var episodes = 0;

            while (!stopRequested)
            {
                var action = MarioControllerEncoder.Decode(state.Controller1, state.Controller2);
                summary.RecordFrame(state, action);

                if (state.IsDead || state.IsLevelComplete || state.ManualResetRequested)
                {
                    episodes++;
                    var reason = state.IsLevelComplete
                        ? MarioTerminalReason.LevelComplete
                        : state.IsDead
                            ? MarioTerminalReason.Death
                            : MarioTerminalReason.ManualReset;
                    recorder.Append(state, action, 0f, done: true, reason);
                    summary.RecordSample();
                    summary.RecordEpisode(state, reason);

                    var reasonLabel = state.IsLevelComplete ? "nivel completado" : state.IsDead ? "muerte" : "reset manual";
                    Console.WriteLine($"    Episodio {episodes} terminado ({reasonLabel}): marioX final {state.MarioX}.");

                    connection.SendCapture(state.LevelIndex);
                    previousX = null;
                    previousCoins = null;
                    previousPowerup = null;
                }
                else
                {
                    var reward = previousX.HasValue ? state.MarioX - previousX.Value : 0f;

                    if (previousCoins.HasValue && state.Coins > previousCoins.Value)
                    {
                        reward += (state.Coins - previousCoins.Value) * SnesEnvironment.CoinReward;
                    }

                    if (previousPowerup.HasValue && state.PowerupLevel != previousPowerup.Value)
                    {
                        var powerupDelta = state.PowerupLevel - previousPowerup.Value;
                        reward += powerupDelta > 0
                            ? powerupDelta * SnesEnvironment.PowerupGainReward
                            : powerupDelta * SnesEnvironment.PowerupLossPenalty;
                    }

                    if (sampler.ShouldRecord(state, action, terminal: false))
                    {
                        recorder.Append(state, action, reward, done: false);
                        summary.RecordSample();
                    }

                    previousX = state.MarioX;
                    previousCoins = state.Coins;
                    previousPowerup = state.PowerupLevel;
                    frames++;

                    connection.SendAction(SnesAction.None);
                }

                if (stopRequested)
                {
                    break;
                }

                state = connection.ReceiveState();

                if (frames > 0 && frames % 600 == 0)
                {
                    Console.WriteLine($"    Capturando... {frames} frames, {episodes} episodios.");
                }
            }

            recorder.Complete();
            summary.Print();
        }

        private static void RunImitationMode()
        {
            Console.WriteLine($"Cargando dataset: {Path.GetFullPath(CapturePath)}");
            var dataset = MarioDatasetLoader.Load(CapturePath);

            if (dataset == null)
            {
                Console.WriteLine("No se pudo leer el dataset; se cancela el entrenamiento de imitacion.");
                return;
            }

            Console.WriteLine($"Dataset: {dataset.Samples.Count} muestras, {dataset.InputCount} entradas, {dataset.OutputCount} salidas.");

            var trainer = new MarioImitationTrainer(dataset, ImitationEpochs);
            var policy = trainer.Train(new Random(1234));
            policy.Save(PolicyPath);

            Console.WriteLine($"Politica guardada: {Path.GetFullPath(PolicyPath)}");
            Console.WriteLine("Usa --play para verla jugar, o --learn (o --seed-policy + NEAT) para seguir mejorandola.");
        }

        private static void RunPlayMode(string[] args)
        {
            Console.WriteLine($"Cargando politica: {Path.GetFullPath(PolicyPath)}");
            var policy = MarioPolicyNetwork.Load(PolicyPath);

            if (policy == null)
            {
                Console.WriteLine(
                    "No se pudo cargar la politica (archivo inexistente, corrupto, o su InputCount/OutputCount " +
                    "no coincide con el encoder/apiedades actuales). Entrena una con --capture + --imitate o --learn.");
                return;
            }

            var levels = ResolveLevels(args);

            using var connection = new SnesBridgeConnection(IPAddress.Parse(Address), BasePort);
            var stopRequested = false;

            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                stopRequested = true;
                Console.WriteLine("\nDeteniendo despues de este episodio... Ctrl+C de nuevo para forzar el cierre inmediato.");
            };

            Console.WriteLine($"Esperando conexion de BizHawk en {Address}:{BasePort}...");
            try
            {
                connection.WaitForBizHawk(() => stopRequested);
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("Se cancelo la espera de conexion con BizHawk.");
                return;
            }
            Console.WriteLine("BizHawk conectado. El modelo entrenado va a jugar solo. Ctrl+C para detener.");

            connection.SendTurbo(true);
            Console.WriteLine("Modo turbo solicitado: el juego corre acelerado.");

            var environment = new SnesEnvironment(connection);
            var episode = 0;

            while (!stopRequested)
            {
                environment.LevelIndex = levels.Length == 0 ? 0 : episode % levels.Length;
                var levelLabel = environment.LevelIndex < levels.Length
                    ? levels[environment.LevelIndex]
                    : environment.LevelIndex.ToString();

                episode++;
                Console.WriteLine($"-- Episodio {episode}: jugando en \"{levelLabel}\" --");

                SnesState state;
                try
                {
                    state = environment.Reset();
                }
                catch (SnesBridgeConnectionLostException ex)
                {
                    Console.WriteLine($"Se perdio la conexion con BizHawk: {ex.Message}");
                    break;
                }

                var stack = new MarioEncoderStack();
                var startX = state.MarioX;
                var steps = 0;
                var done = false;

                while (!done && !stopRequested)
                {
                    var input = stack.Encode(state);
                    var output = policy.Forward(input);
                    var action = MarioAgentOutput.ToAction(output);

                    EnvironmentStepResult<SnesState> result;
                    try
                    {
                        result = environment.Step(action);
                    }
                    catch (SnesBridgeConnectionLostException ex)
                    {
                        Console.WriteLine($"Se perdio la conexion con BizHawk: {ex.Message}");
                        stopRequested = true;
                        break;
                    }

                    state = result.State;
                    done = result.Done;
                    steps++;
                }

                Console.WriteLine($"    Episodio {episode} terminado: {steps} frames, avance {state.MarioX - startX} px.");
            }

            Console.WriteLine("Reproduccion detenida.");
        }

        private static void RunEvaluateMode(string[] args)
        {
            var modelPath = GetOption(args, "--model") ?? BestModelPath;
            Console.WriteLine($"Cargando modelo: {Path.GetFullPath(modelPath)}");
            var genome = MarioNeatModelStore.Load(modelPath);

            if (genome == null)
            {
                Console.WriteLine(
                    "No se pudo cargar el modelo (archivo inexistente, corrupto, o su InputCount/OutputCount " +
                    "no coincide con el encoder actual). Un modelo valido se genera con el modo evolutivo " +
                    $"(deja el mejor genoma en {BestModelPath} en cada generacion), o pasa --model <ruta> a otro .navm de genoma NEAT.");
                return;
            }

            var levels = ResolveLevels(args);
            var episodesPerLevel = Math.Max(1, TryGetInt(GetOption(args, "--episodes"), DefaultEvaluationEpisodes));
            var agent = new MarioAgent(genome);
            var report = new MarioGeneralizationReport();

            using var connection = new SnesBridgeConnection(IPAddress.Parse(Address), BasePort);
            var stopRequested = false;

            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                stopRequested = true;
                Console.WriteLine("\nDeteniendo evaluacion (el episodio en curso no se cuenta en el reporte)...");
            };

            Console.WriteLine($"Esperando conexion de BizHawk en {Address}:{BasePort}...");
            try
            {
                connection.WaitForBizHawk(() => stopRequested);
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("Se cancelo la espera de conexion con BizHawk.");
                return;
            }

            Console.WriteLine(
                $"BizHawk conectado. Evaluando {episodesPerLevel} episodio(s) por nivel en {levels.Length} nivel(es), " +
                "SIN entrenar ni modificar el modelo. Ctrl+C para cortar antes de tiempo.");

            connection.SendTurbo(true);
            var environment = new SnesEnvironment(connection);

            for (var levelIndex = 0; levelIndex < levels.Length && !stopRequested; levelIndex++)
            {
                environment.LevelIndex = levelIndex;
                var levelLabel = levels[levelIndex];

                for (var episode = 0; episode < episodesPerLevel && !stopRequested; episode++)
                {
                    Console.WriteLine($"-- Nivel \"{levelLabel}\": episodio {episode + 1}/{episodesPerLevel} --");

                    SnesState state;
                    try
                    {
                        state = environment.Reset();
                    }
                    catch (SnesBridgeConnectionLostException ex)
                    {
                        Console.WriteLine($"Se perdio la conexion con BizHawk: {ex.Message}");
                        stopRequested = true;
                        break;
                    }

                    agent.ResetHistory();
                    var bestX = state.MarioX;
                    var previousState = state;
                    var steps = 0;
                    var done = false;

                    while (!done && !stopRequested && steps < MaxStepsPerEpisode)
                    {
                        previousState = state;
                        var action = agent.Decide(state);

                        EnvironmentStepResult<SnesState> result;
                        try
                        {
                            result = environment.Step(action);
                        }
                        catch (SnesBridgeConnectionLostException ex)
                        {
                            Console.WriteLine($"Se perdio la conexion con BizHawk: {ex.Message}");
                            stopRequested = true;
                            break;
                        }

                        state = result.State;
                        done = result.Done;
                        steps++;

                        if (state.MarioX > bestX)
                        {
                            bestX = state.MarioX;
                        }
                    }

                    if (stopRequested && !done)
                    {
                        // Episodio cortado a mano (Ctrl+C): no lo contamos,
                        // para no ensuciar el reporte con una corrida
                        // incompleta.
                        break;
                    }

                    var completed = state.IsLevelComplete;
                    var deathCause = state.IsDead ? MarioFitnessEvaluator.ClassifyDeathCause(previousState) : MarioDeathCause.None;
                    var stepsCapReached = !completed && deathCause == MarioDeathCause.None && steps >= MaxStepsPerEpisode;

                    report.RecordEpisode(levelIndex, bestX, completed, deathCause, stepsCapReached);

                    var outcome = completed
                        ? "nivel completado"
                        : deathCause == MarioDeathCause.Enemy
                            ? "murio (enemigo)"
                            : deathCause == MarioDeathCause.FallOrHazard
                                ? "murio (caida/peligro)"
                                : stepsCapReached
                                    ? "tope de pasos sin morir ni completar"
                                    : "reset manual";

                    Console.WriteLine($"    Episodio terminado: {steps} frames, bestX {bestX}, {outcome}.");
                }
            }

            connection.SendTurbo(false);

            Console.WriteLine();
            Console.WriteLine("--- Resultado de evaluacion (sin entrenar) ---");
            report.PrintTo(Console.WriteLine, levels);

            if (report.ByLevel.Count > 0)
            {
                var historyFileExists = File.Exists(GeneralizationReportPath);
                using (var writer = new StreamWriter(GeneralizationReportPath, append: historyFileExists))
                {
                    if (!historyFileExists)
                    {
                        writer.WriteLine("timestamp_utc,model,level,episodes,completions,completion_pct,avg_best_x,best_x_max,deaths_enemy,deaths_fall,steps_cap");
                    }

                    foreach (var row in report.ToCsvRows(DateTime.UtcNow, modelPath, levels))
                    {
                        writer.WriteLine(row);
                    }
                }

                Console.WriteLine($"Reporte guardado (append) en {Path.GetFullPath(GeneralizationReportPath)}.");
            }
            else
            {
                Console.WriteLine("No se completo ningun episodio; no se escribio el reporte.");
            }
        }

        private static void RunLearnMode(string[] args)
        {
            var levels = ResolveLevels(args);

            using var connection = new SnesBridgeConnection(IPAddress.Parse(Address), BasePort);

            var forceExit = false;
            var advance = false;
            DateTime? lastCancel = null;

            Console.CancelKeyPress += (_, e) =>
            {
                e.Cancel = true;
                var now = DateTime.UtcNow;
                var doublePress = lastCancel.HasValue && (now - lastCancel.Value).TotalMilliseconds < 1200;
                lastCancel = now;

                if (doublePress)
                {
                    forceExit = true;
                    Console.WriteLine("\nSalida forzada.");
                }
                else
                {
                    advance = true;
                    Console.WriteLine("\nAvanzando de fase... (Ctrl+C doble y rapido para salir del todo).");
                }
            };

            Console.WriteLine($"Esperando conexion de BizHawk en {Address}:{BasePort}...");
            try
            {
                connection.WaitForBizHawk(() => forceExit);
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("Se cancelo la espera de conexion con BizHawk. No se grabo ni entreno nada.");
                return;
            }

            var iteration = 0;

            while (!forceExit)
            {
                advance = false;
                iteration++;
                var frames = 0;
                var episodes = 0;

                Console.WriteLine($"--- PASO 1/{iteration}: captura (juga con el teclado) ---");
                Console.WriteLine("Muertes, fin de nivel y el reset manual (tecla Insert) cierran el episodio.");
                Console.WriteLine("Ctrl+C cuando hayas jugado: se entrena con todo lo acumulado y el modelo juega solo.");

                using var recorder = new MarioDatasetRecorder(CapturePath, append: true);
                var sampler = new MarioFrameSampler();
                var summary = new MarioCaptureSummary();
                var state = connection.ReceiveState();
                int? previousX = null;
                int? previousCoins = null;
                int? previousPowerup = null;

                while (!advance && !forceExit)
                {
                    var action = MarioControllerEncoder.Decode(state.Controller1, state.Controller2);
                    summary.RecordFrame(state, action);

                    if (state.IsDead || state.IsLevelComplete || state.ManualResetRequested)
                    {
                        episodes++;
                        var reason = state.IsLevelComplete
                            ? MarioTerminalReason.LevelComplete
                            : state.IsDead
                                ? MarioTerminalReason.Death
                                : MarioTerminalReason.ManualReset;
                        recorder.Append(state, action, 0f, done: true, reason);
                        summary.RecordSample();
                        summary.RecordEpisode(state, reason);

                        var reasonLabel = state.IsLevelComplete ? "nivel completado" : state.IsDead ? "muerte" : "reset manual";
                        Console.WriteLine($"    Episodio {episodes} terminado ({reasonLabel}): marioX final {state.MarioX}.");

                        connection.SendCapture(state.LevelIndex);
                        previousX = null;
                        previousCoins = null;
                        previousPowerup = null;
                    }
                    else
                    {
                        var reward = previousX.HasValue ? state.MarioX - previousX.Value : 0f;

                        if (previousCoins.HasValue && state.Coins > previousCoins.Value)
                        {
                            reward += (state.Coins - previousCoins.Value) * SnesEnvironment.CoinReward;
                        }

                        if (previousPowerup.HasValue && state.PowerupLevel != previousPowerup.Value)
                        {
                            var powerupDelta = state.PowerupLevel - previousPowerup.Value;
                            reward += powerupDelta > 0
                                ? powerupDelta * SnesEnvironment.PowerupGainReward
                                : powerupDelta * SnesEnvironment.PowerupLossPenalty;
                        }

                        if (sampler.ShouldRecord(state, action, terminal: false))
                        {
                            recorder.Append(state, action, reward, done: false);
                            summary.RecordSample();
                        }

                        previousX = state.MarioX;
                        previousCoins = state.Coins;
                        previousPowerup = state.PowerupLevel;
                        frames++;

                        connection.SendAction(SnesAction.None);
                    }

                    if (advance || forceExit)
                    {
                        break;
                    }

                    state = connection.ReceiveState();

                    if (frames > 0 && frames % 600 == 0)
                    {
                        Console.WriteLine($"    Capturando... {frames} frames, {episodes} episodios.");
                    }
                }

                recorder.Complete();
                summary.Print();

                if (forceExit)
                {
                    break;
                }

                Console.WriteLine($"--- PASO 2/{iteration}: entrenando con todo lo grabado ({ImitationEpochs} epochs) ---");
                var dataset = MarioDatasetLoader.Load(CapturePath);
                if (dataset == null || dataset.Samples.Count == 0)
                {
                    Console.WriteLine("No hay muestras para entrenar; termina el bucle.");
                    break;
                }

                Console.WriteLine($"Dataset acumulado: {dataset.Samples.Count} muestras, {dataset.InputCount} entradas, {dataset.OutputCount} salidas.");

                var previousPolicy = MarioPolicyNetwork.Load(PolicyPath);
                if (previousPolicy != null)
                {
                    Console.WriteLine("Continuando el entrenamiento desde la politica anterior.");
                }

                var trainer = new MarioImitationTrainer(dataset, ImitationEpochs);
                var policy = trainer.Train(new Random(1234), previousPolicy);
                policy.Save(PolicyPath);

                Console.WriteLine($"Politica actualizada: {Path.GetFullPath(PolicyPath)}");

                if (forceExit)
                {
                    break;
                }

                Console.WriteLine($"--- PASO 3/{iteration}: el modelo juega solo (Ctrl+C para volver a capturar) ---");
                advance = false;

                var environment = new SnesEnvironment(connection);
                environment.MarkConnected();
                connection.SendTurbo(true);

                var episode = 0;

                while (!advance && !forceExit)
                {
                    environment.LevelIndex = levels.Length == 0 ? 0 : episode % levels.Length;
                    var levelLabel = environment.LevelIndex < levels.Length
                        ? levels[environment.LevelIndex]
                        : environment.LevelIndex.ToString();

                    episode++;
                    Console.WriteLine($"-- Episodio {episode}: jugando en \"{levelLabel}\" --");

                    SnesState playState;
                    try
                    {
                        playState = environment.Reset();
                    }
                    catch (SnesBridgeConnectionLostException ex)
                    {
                        Console.WriteLine($"Se perdio la conexion con BizHawk: {ex.Message}");
                        forceExit = true;
                        break;
                    }

                    var stack = new MarioEncoderStack();
                    var startX = playState.MarioX;
                    var steps = 0;
                    var done = false;

                    while (!done && !advance && !forceExit)
                    {
                        var input = stack.Encode(playState);
                        var output = policy.Forward(input);
                        var action = MarioAgentOutput.ToAction(output);

                        EnvironmentStepResult<SnesState> result;
                        try
                        {
                            result = environment.Step(action);
                        }
                        catch (SnesBridgeConnectionLostException ex)
                        {
                            Console.WriteLine($"Se perdio la conexion con BizHawk: {ex.Message}");
                            forceExit = true;
                            break;
                        }

                        playState = result.State;
                        done = result.Done;
                        steps++;
                    }

                    Console.WriteLine($"    Episodio {episode} terminado: {steps} frames, avance {playState.MarioX - startX} px.");
                }

                connection.SendTurbo(false);
                Console.WriteLine("Reproduccion detenida; volviendo a capturar. Cuanto mejor juegue el modelo, mas controles muestra el dataset acumulado.");
            }

            Console.WriteLine("Bucle terminado.");
        }

        private static void RunTrainingModeCore(string[] args, int populationSize, int workers, bool freshRequested, SnesBridgeConnection[] connections)
        {
            var seedPolicyPath = GetOption(args, "--seed-policy");
            var levels = ResolveLevels(args);
            var curriculumEnabled = args.Any(arg => arg.Equals("--curriculum", StringComparison.OrdinalIgnoreCase));
            var curriculumWindow = TryGetInt(GetOption(args, "--curriculum-window"), MarioCurriculum.DefaultWindowSize);
            var curriculumThreshold = TryGetFloat(GetOption(args, "--curriculum-threshold"), MarioCurriculum.DefaultAdvanceThresholdPercent);

            if (freshRequested)
            {
                MarioCheckpointStore.Delete();
                DeleteIfExists(FitnessHistoryPath);
                Console.WriteLine("Progreso anterior eliminado, arrancando desde cero.");
            }

            Console.WriteLine($"Archivo de entrenamiento: {Path.GetFullPath(CheckpointPath)}");

            var stopRequested = false;
            var forceImmediate = false;

            Console.CancelKeyPress += (_, e) =>
            {
                if (forceImmediate)
                {
                    Console.WriteLine("\nForzando cierre inmediato. El progreso de la generacion en curso puede perderse (el ultimo checkpoint guardado sigue intacto).");
                    return;
                }

                e.Cancel = true;
                stopRequested = true;
                forceImmediate = true;
                Console.WriteLine("\nSaliendo despues de esta generacion... puede tardar si esta esperando datos de BizHawk. Presiona Ctrl+C de nuevo para forzar el cierre inmediato.");
            };

            if (workers == 1)
            {
                Console.WriteLine($"Esperando conexion de BizHawk en {Address}:{BasePort}...");
                try
                {
                    connections[0].WaitForBizHawk(() => stopRequested);
                }
                catch (OperationCanceledException)
                {
                    Console.WriteLine("Se cancelo la espera de conexion con BizHawk.");
                    return;
                }
            }
            else
            {
                for (var w = 0; w < workers; w++)
                {
                    Console.WriteLine($"Esperando conexion de BizHawk worker {w + 1}/{workers} en {Address}:{BasePort + w}...");
                    try
                    {
                        connections[w].WaitForBizHawk(() => stopRequested);
                    }
                    catch (OperationCanceledException)
                    {
                        Console.WriteLine("Se cancelo la espera de conexion con BizHawk.");
                        return;
                    }
                }

                Console.WriteLine($"Cada worker usa su propia instancia de BizHawk/Lua en el puerto correspondiente (NEURAVAL_PORT=8766..{BasePort + workers - 1}).");
            }

            Console.WriteLine("BizHawk conectado.");
            foreach (var connection in connections)
            {
                connection.SendTurbo(true);
            }
            Console.WriteLine("Modo turbo solicitado: BizHawk deberia acelerar a la maxima velocidad soportada.");

            var environments = new SnesEnvironment[workers];
            for (var w = 0; w < workers; w++)
            {
                environments[w] = new SnesEnvironment(connections[w]);
            }

            var environment = environments[0];
            var random = new Random(1234);
            var tracker = new NeatInnovationTracker(MarioAgent.InputCount + MarioAgent.OutputCount + 1);

            var checkpoint = MarioCheckpointStore.Load();

            MarioAgent[] initialAgents;
            int startingGeneration;
            var bestFitnessEver = 0f;
            NeatGenome? bestGenomeEver = null;

            if (checkpoint != null && checkpoint.Genomes.Count > 0)
            {
                var maxNodeId = checkpoint.Genomes.SelectMany(genome => genome.Nodes).Max(node => node.Id);
                var maxInnovation = checkpoint.Genomes
                    .SelectMany(genome => genome.Connections)
                    .Select(gene => (int?)gene.Innovation)
                    .DefaultIfEmpty()
                    .Max() ?? -1;

                tracker.FastForwardTo(maxNodeId + 1, maxInnovation + 1);

                initialAgents = checkpoint.Genomes.Select(genome => new MarioAgent(genome)).ToArray();
                startingGeneration = checkpoint.Generation;
                bestFitnessEver = checkpoint.BestFitnessEver;
                bestGenomeEver = checkpoint.BestGenomeEver;

                Console.WriteLine($"Progreso anterior encontrado: reanudando desde la generacion {startingGeneration}.");
            }
            else
            {
                var seedGenome = TryLoadSeedGenome(seedPolicyPath, tracker);

                initialAgents = new MarioAgent[populationSize];
                for (var i = 0; i < populationSize; i++)
                {
                    initialAgents[i] = seedGenome != null && i == 0
                        ? new MarioAgent(seedGenome.Clone())
                        : MarioAgent.CreateRandom(random, tracker);
                }

                startingGeneration = 0;
            }

            MarioCurriculum? curriculum = null;
            if (curriculumEnabled)
            {
                curriculum = new MarioCurriculum(
                    levels.Length,
                    curriculumWindow,
                    curriculumThreshold,
                    checkpoint?.CurriculumStageIndex ?? 0,
                    checkpoint?.CurriculumGenerationsAtStage ?? 0);

                Console.WriteLine(
                    $"Curriculum activado: {levels.Length} tramo(s) via --level, ventana de {curriculumWindow} " +
                    $"generaciones, umbral {curriculumThreshold:F1}% de completions para avanzar de tramo. " +
                    $"Arranca en el tramo {curriculum.StageIndex + 1}/{levels.Length}" +
                    (checkpoint != null ? " (restaurado del checkpoint)." : "."));

                if (levels.Length <= 1)
                {
                    Console.WriteLine(
                        "Aviso: --curriculum esta activo pero solo hay un --level configurado (o ninguno), asi " +
                        "que no hay a donde avanzar; se comporta igual que sin --curriculum. Agrega mas --level " +
                        "(uno por savestate de cada tramo) para que tenga efecto.");
                }
            }

            var options = new NeatEvolutionOptions();
            var strategy = new NeatEvolutionStrategy<MarioAgent>(random, tracker, options, genome => new MarioAgent(genome));

            var evaluator = new MarioFitnessEvaluator(MaxStepsPerEpisode);

            var population = new Population<MarioAgent, SnesState, SnesAction>(
                initialAgents,
                () => environment,
                evaluator,
                strategy);

            Console.WriteLine("Neuraval.Evolution.MarioBridge");
            Console.WriteLine("Generacion | Especies | Fitness promedio | Mejor fitness | Best X | % completado");

            var historyFileExists = File.Exists(FitnessHistoryPath);
            using var historyWriter = new StreamWriter(FitnessHistoryPath, append: historyFileExists && !freshRequested);
            if (!historyFileExists || freshRequested)
            {
                historyWriter.WriteLine("generation,level,species,avg_fitness,best_fitness,best_ever,best_x,attempts,completions,completion_pct,deaths_enemy,deaths_fall,steps_cap,curriculum_stage,curriculum_window_avg_pct");
                historyWriter.Flush();
            }

            var generation = startingGeneration;
            var stagnantGenerations = 0;
            var stagnationEscalated = false;
            var maxConsecutiveConnectionFailures = 5;
            var consecutiveConnectionFailures = 0;

            while (!stopRequested)
            {
                generation++;

                var currentLevelIndex = curriculum != null
                    ? curriculum.StageIndex
                    : (levels.Length == 0 ? 0 : generation % levels.Length);

                for (var w = 0; w < workers; w++)
                {
                    environments[w].LevelIndex = currentLevelIndex;
                }

                var currentLevelLabel = currentLevelIndex < levels.Length
                    ? levels[currentLevelIndex]
                    : currentLevelIndex.ToString();

                if (curriculum != null)
                {
                    Console.WriteLine(
                        $"-- Generacion {generation}: curriculum tramo {curriculum.StageIndex + 1}/{levels.Length} " +
                        $"\"{currentLevelLabel}\" (gen {curriculum.GenerationsAtStage} en este tramo, ventana " +
                        $"{curriculum.WindowSampleCount}/{curriculum.WindowCapacity} @ {curriculum.WindowAverageCompletion:F1}% completions) --");
                }
                else
                {
                    for (var w = 0; w < workers; w++)
                    {
                        Console.WriteLine($"-- Generacion {generation}: worker {w + 1} en \"{currentLevelLabel}\" --");
                    }
                }

                IReadOnlyList<float> scores;
                try
                {
                    scores = workers > 1
                        ? population.EvaluateGenerationParallel(w => environments[w], workers)
                        : population.EvaluateGeneration();
                }
                catch (SnesBridgeConnectionLostException ex)
                {
                    consecutiveConnectionFailures++;
                    Console.WriteLine($"Se corto la conexion con BizHawk durante la generacion {generation}: {ex.Message}");

                    if (consecutiveConnectionFailures > maxConsecutiveConnectionFailures)
                    {
                        Console.WriteLine(
                            $"Se perdio la conexion {consecutiveConnectionFailures} veces seguidas sin llegar a completar " +
                            "una generacion; esto ya no parece un reinicio puntual de BizHawk. Se detiene el entrenamiento. " +
                            "El ultimo checkpoint guardado sigue intacto.");
                        break;
                    }

                    var reconnectFailed = false;
                    for (var w = 0; w < workers; w++)
                    {
                        try
                        {
                            connections[w].Reconnect(() => stopRequested);
                        }
                        catch (OperationCanceledException)
                        {
                            Console.WriteLine("Se cancelo la espera de reconexion con BizHawk.");
                            reconnectFailed = true;
                            break;
                        }

                        environments[w].NotifyReconnected();
                        environments[w].LevelIndex = currentLevelIndex;
                    }

                    if (reconnectFailed)
                    {
                        break;
                    }

                    generation--;
                    continue;
                }

                consecutiveConnectionFailures = 0;
                var bestThisGeneration = scores.Count == 0 ? 0f : scores.Max();

                if (bestThisGeneration > bestFitnessEver || bestGenomeEver == null)
                {
                    var bestIndex = scores.ToList().IndexOf(bestThisGeneration);
                    bestFitnessEver = bestThisGeneration;
                    bestGenomeEver = population.Agents[bestIndex].Genome.Clone();
                    stagnantGenerations = 0;
                }
                else
                {
                    stagnantGenerations++;
                    if (stagnantGenerations >= StagnationWatchGenerations && !stagnationEscalated)
                    {
                        options.WeightPerturbStrength += 0.5f;
                        options.AddConnectionRate += 0.04f;
                        options.AddNodeRate += 0.02f;
                        options.CompatibilityThreshold += 1f;
                        stagnationEscalated = true;
                        Console.WriteLine($"Sin mejora en las ultimas {StagnationWatchGenerations} generaciones: se aumenta la presion exploratoria (mutaciones mas fuertes y mas nodos/conexiones).");
                    }
                }

                var attempts = scores.Count;
                var completionPct = attempts == 0 ? 0f : evaluator.Completions * 100f / attempts;

                strategy.GlobalBestGenome = bestGenomeEver;
                population.Advance();
                Console.WriteLine($"{generation,10} | {strategy.LastSpeciesCount,8} | {population.AverageFitness,17:F2} | {population.BestFitness,13:F2} | {evaluator.BestXThisGeneration,7} | {completionPct,11:F1}%");

                var levelSample = environments[0].LevelIndex;
                var levelLabel = levelSample < levels.Length ? levels[levelSample] : levelSample.ToString();

                if (curriculum != null)
                {
                    var advancedStage = curriculum.RecordGeneration(completionPct);
                    if (advancedStage)
                    {
                        var newLabel = curriculum.StageIndex < levels.Length
                            ? levels[curriculum.StageIndex]
                            : curriculum.StageIndex.ToString();
                        Console.WriteLine(
                            $"Curriculum: avance de tramo -> {curriculum.StageIndex + 1}/{levels.Length} \"{newLabel}\" " +
                            $"(promedio de completions en la ventana >= {curriculumThreshold:F1}%).");
                    }
                    else if (curriculum.IsAtFinalStage)
                    {
                        Console.WriteLine($"Curriculum: sostenido en el ultimo tramo \"{levelLabel}\" (no hay mas --level configurados).");
                    }
                }

                historyWriter.WriteLine(
                    string.Join(",",
                        generation,
                        CsvQuote(levelLabel),
                        strategy.LastSpeciesCount,
                        population.AverageFitness.ToString("F2", CultureInfo.InvariantCulture),
                        population.BestFitness.ToString("F2", CultureInfo.InvariantCulture),
                        bestFitnessEver.ToString("F2", CultureInfo.InvariantCulture),
                        evaluator.BestXThisGeneration,
                        attempts,
                        evaluator.Completions,
                        completionPct.ToString("F1", CultureInfo.InvariantCulture),
                        evaluator.DeathsByEnemy,
                        evaluator.DeathsByFall,
                        evaluator.StepsCapTerminations,
                        curriculum != null ? (curriculum.StageIndex + 1).ToString(CultureInfo.InvariantCulture) : string.Empty,
                        curriculum != null ? curriculum.WindowAverageCompletion.ToString("F1", CultureInfo.InvariantCulture) : string.Empty));
                historyWriter.Flush();
                evaluator.ResetCounters();

                MarioCheckpointStore.Save(new MarioCheckpoint
                {
                    Generation = generation,
                    BestFitnessEver = bestFitnessEver,
                    BestGenomeEver = bestGenomeEver,
                    Genomes = population.Agents.Select(agent => agent.Genome).ToList(),
                    CurriculumStageIndex = curriculum?.StageIndex,
                    CurriculumGenerationsAtStage = curriculum?.GenerationsAtStage
                });

                if (bestGenomeEver != null)
                {
                    MarioNeatModelStore.Save(BestModelPath, bestGenomeEver);
                }
            }

            Console.WriteLine("Entrenamiento detenido por el usuario. Progreso guardado.");
        }

        private static string CsvQuote(string value)
        {
            if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n'))
            {
                return value;
            }

            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        private static void DeleteIfExists(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
            catch
            {
            }
        }

        private static NeatGenome? TryLoadSeedGenome(string? seedPolicyPath, NeatInnovationTracker tracker)
        {
            if (seedPolicyPath == null)
            {
                return null;
            }

            var policy = MarioPolicyNetwork.Load(seedPolicyPath);
            if (policy == null)
            {
                Console.WriteLine($"No se pudo cargar la politica semilla en {Path.GetFullPath(seedPolicyPath)}; se arranca con genomas random.");
                return null;
            }

            Console.WriteLine($"Sembrando poblacion con politica imitada: {Path.GetFullPath(seedPolicyPath)}.");

            var genome = policy.AsGenome(tracker);
            var maxNodeId = genome.Nodes.Max(node => node.Id);
            var maxInnovation = genome.Connections.Count - 1;
            tracker.FastForwardTo(maxNodeId + 1, maxInnovation + 1);
            return genome;
        }

        private static string[] ResolveLevels(string[] args)
        {
            var levels = new List<string>();

            for (var i = 0; i + 1 < args.Length; i++)
            {
                if (args[i].Equals("--level", StringComparison.OrdinalIgnoreCase))
                {
                    levels.Add(args[i + 1]);
                    i++;
                }
            }

            return levels.Count > 0 ? levels.ToArray() : (string[])DefaultLevels.Clone();
        }

        private static string? GetOption(string[] args, string name)
        {
            var index = Array.FindIndex(args, arg => arg.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (index >= 0 && index + 1 < args.Length)
            {
                return args[index + 1];
            }

            return null;
        }

        private static int TryGetInt(string? value, int fallback)
        {
            return int.TryParse(value, out var parsed) ? parsed : fallback;
        }

        private static float TryGetFloat(string? value, float fallback)
        {
            return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : fallback;
        }
    }
}