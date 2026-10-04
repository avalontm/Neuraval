using Neuraval.Samples.DinoGame.Sources;
using Neuraval.Samples.DinoGame.Sources.UI;
using Neuraval.Evolution;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Samples.DinoGame
{
    public class MainGame : Game
    {
        public static GraphicsDeviceManager _graphics;
        public static IList<Dino> players;
        public static IList<BaseEnemy> enemies;
        public static GameTime time { private set; get; }
        public const int SpawnX = 1350;
        float speedStart = 12;
        public static float speed = 12;
        public static bool GameOver = false;
        public static Dino LastPlayer;
        public static bool IsDebug { private set; get; }

        bool showBrainViz = true;
        SpriteBatch _spriteBatch;
        StageBackground background;
        SpriteFont font;

        GameState state = GameState.MainMenu;
        GameState stateBeforeOptions = GameState.MainMenu;
        Menu mainMenu;
        Menu optionsMenu;
        Menu pauseMenu;
        Menu confirmNewTrainingMenu;
        DinoEvolutionSaveData pendingSaveData;

        readonly DinoTrainingSettings settings;

        static readonly int[] PopulationSizeOptions = { 100, 250, 500, 1000, 1500, 2000, 3000, 5000 };

        readonly Random evolutionRandom = new Random();
        readonly ElitistMutationStrategy<NeuralNetwork> evolutionStrategy;

        float bestFitnessEver = 0f;
        DinoGenome bestGenomeEver = null;

        ProbabilidadPorcentaje probabilidad;
        int every_sec = 0;
        int spawnIntervalFrames;
        int generation = 0;
        int alive = 0;
        Dino playerTarget;

        double roundElapsedSeconds = 0d;

        public MainGame()
        {
            _graphics = new GraphicsDeviceManager(this);
            _graphics.PreferredBackBufferHeight = 720;
            _graphics.PreferredBackBufferWidth = 1280;
            _graphics.SynchronizeWithVerticalRetrace = false;

            this.IsFixedTimeStep = true;
            this.TargetElapsedTime = TimeSpan.FromSeconds(1d / 60d);

            Content.RootDirectory = "Content";
            IsMouseVisible = true;

            settings = DinoTrainingSettings.Load();
            speedStart = settings.SpeedStart;
            spawnIntervalFrames = settings.EnemySpawnIntervalFrames;
            DinoStartX = settings.DinoStartX;
            DinoStartXJitter = settings.DinoStartXJitter;

            evolutionStrategy = new ElitistMutationStrategy<NeuralNetwork>(
                evolutionRandom, settings.EliteCount, settings.MutationRate, NeuralNetwork.MutationStrength,
                settings.RandomInjectionFraction, () => new NeuralNetwork());
        }

        public static int DinoStartX = 200;
        public static int DinoStartXJitter = 80;

        protected override void Initialize()
        {
            base.Initialize();
        }

        protected override void LoadContent()
        {
            _spriteBatch = new SpriteBatch(GraphicsDevice);

            font = Content.Load<SpriteFont>("default");

            DrawManager.Init(_graphics.GraphicsDevice);
            probabilidad = new ProbabilidadPorcentaje();
            Animations.Load(Content);

            background = new StageBackground();
            players = new List<Dino>();
            enemies = new List<BaseEnemy>();

            pendingSaveData = DinoEvolutionStore.Load();

            BuildMenus();
        }

        void BuildMenus()
        {
            BuildMainMenu();

            optionsMenu = new Menu(new[]
            {
                new MenuItem(PopulationSizeLabel, () => AdjustPopulation(1), AdjustPopulation),
                new MenuItem(() => $"Pantalla completa: {(_graphics.IsFullScreen ? "ON" : "OFF")}", ToggleFullscreen),
                new MenuItem(() => $"HUD de depuracion: {(IsDebug ? "ON" : "OFF")}", ToggleDebugHud),
                new MenuItem(() => $"Red neuronal: {(showBrainViz ? "ON" : "OFF")}", ToggleBrainViz),
                new MenuItem("Volver", CloseOptions)
            });

            pauseMenu = new Menu(new[]
            {
                new MenuItem("Continuar", ResumeGame),
                new MenuItem("Opciones", OpenOptions),
                new MenuItem("Menu Principal", GoToMainMenu),
                new MenuItem("Salir", () => Exit())
            });

            confirmNewTrainingMenu = new Menu(new[]
            {
                new MenuItem("Si, borrar y empezar de cero", ConfirmNewTraining),
                new MenuItem("Cancelar", CancelNewTraining)
            });
        }

        void BuildMainMenu()
        {
            mainMenu = new Menu(BuildMainMenuItems());
        }

        IEnumerable<MenuItem> BuildMainMenuItems()
        {
            if (CanContinueTraining())
            {
                yield return new MenuItem("Continuar Entrenamiento", ContinueTraining);
            }

            yield return new MenuItem("Nuevo Entrenamiento", OpenConfirmNewTraining);
            yield return new MenuItem("Opciones", OpenOptions);
            yield return new MenuItem("Salir", () => Exit());
        }

        bool CanContinueTraining()
        {
            return players.Count > 0 || pendingSaveData != null;
        }

        void ContinueTraining()
        {
            if (players.Count == 0 && pendingSaveData != null)
            {
                generation = pendingSaveData.Generation;
                bestFitnessEver = pendingSaveData.BestFitnessEver;
                bestGenomeEver = pendingSaveData.BestGenomeEver;

                var brains = DinoEvolutionStore.RebuildPopulation(
                    pendingSaveData, settings.PopulationSize, evolutionRandom, settings.MutationRate, NeuralNetwork.MutationStrength,
                    settings.RandomInjectionFraction);

                SetupPopulation(brains);
            }

            PrepareRound();
            state = GameState.Playing;
        }

        void OpenConfirmNewTraining()
        {
            confirmNewTrainingMenu.Reset();
            state = GameState.ConfirmNewTraining;
        }

        void CancelNewTraining()
        {
            GoToMainMenu();
        }

        void ConfirmNewTraining()
        {
            DinoEvolutionStore.Delete();
            pendingSaveData = null;

            generation = 0;
            bestFitnessEver = 0f;
            bestGenomeEver = null;

            var brains = new List<NeuralNetwork>(settings.PopulationSize);

            for (int i = 0; i < settings.PopulationSize; i++)
            {
                brains.Add(new NeuralNetwork());
            }

            SetupPopulation(brains);

            PrepareRound();
            state = GameState.Playing;
        }

        void SetupPopulation(List<NeuralNetwork> brains)
        {
            players.Clear();

            foreach (var brain in brains)
            {
                players.Add(new Dino(brain));
            }

            alive = players.Count;
        }

        void PrepareRound()
        {
            enemies.Clear();
            every_sec = 0;
            spawnIntervalFrames = settings.EnemySpawnIntervalFrames;
            speed = speedStart;
            roundElapsedSeconds = 0d;
            GameOver = false;
        }

        void OpenOptions()
        {
            stateBeforeOptions = state;
            optionsMenu.Reset();
            state = GameState.Options;
        }

        void CloseOptions()
        {
            state = stateBeforeOptions;
        }

        void PauseGame()
        {
            pauseMenu.Reset();
            state = GameState.Paused;
        }

        void ResumeGame()
        {
            state = GameState.Playing;
        }

        void GoToMainMenu()
        {
            BuildMainMenu();
            mainMenu.Reset();
            state = GameState.MainMenu;
        }

        void ToggleFullscreen()
        {
            _graphics.IsFullScreen = !_graphics.IsFullScreen;
            _graphics.ApplyChanges();
        }

        void ToggleDebugHud()
        {
            IsDebug = !IsDebug;
        }

        void ToggleBrainViz()
        {
            showBrainViz = !showBrainViz;
        }

        string PopulationSizeLabel()
        {
            string pending = players.Count > 0 && players.Count != settings.PopulationSize ? "  (se aplica al reiniciar)" : string.Empty;
            return $"Numero de dinos: {settings.PopulationSize}{pending}";
        }

        void AdjustPopulation(int direction)
        {
            int index = Array.IndexOf(PopulationSizeOptions, settings.PopulationSize);

            if (index < 0)
            {
                index = 0;

                for (int i = 1; i < PopulationSizeOptions.Length; i++)
                {
                    if (PopulationSizeOptions[i] <= settings.PopulationSize)
                    {
                        index = i;
                    }
                }
            }

            index = Math.Clamp(index + direction, 0, PopulationSizeOptions.Length - 1);
            settings.PopulationSize = PopulationSizeOptions[index];
            DinoTrainingSettings.Save(settings);
        }

        protected override void Update(GameTime gameTime)
        {
            base.Update(gameTime);
            time = gameTime;

            InputManager.GetState();

            switch (state)
            {
                case GameState.MainMenu:
                    mainMenu.Update();
                    break;

                case GameState.Options:
                    optionsMenu.Update();
                    if (InputManager.IsKeyPressed(Keys.Escape, true))
                    {
                        CloseOptions();
                    }
                    break;

                case GameState.ConfirmNewTraining:
                    confirmNewTrainingMenu.Update();
                    if (InputManager.IsKeyPressed(Keys.Escape, true))
                    {
                        CancelNewTraining();
                    }
                    break;

                case GameState.Paused:
                    pauseMenu.Update();
                    if (InputManager.IsKeyPressed(Keys.Escape, true))
                    {
                        ResumeGame();
                    }
                    break;

                case GameState.Playing:
                    UpdateGameplay(gameTime);
                    break;
            }
        }

        void UpdateGameplay(GameTime gameTime)
        {
            if (GameOver)
            {
                if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed
                    || Keyboard.GetState().IsKeyDown(Keys.Escape)
                    || Keyboard.GetState().IsKeyDown(Keys.Enter)
                    || Keyboard.GetState().IsKeyDown(Keys.Space))
                {
                    GameStart();
                }

                return;
            }

            if (InputManager.IsKeyPressed(Keys.Escape, true))
            {
                PauseGame();
                return;
            }

            if (InputManager.IsKeyPressed( Keys.F1, true))
            {
                IsDebug = !IsDebug;
            }

            if (InputManager.IsKeyPressed(Keys.F2, true))
            {
                showBrainViz = !showBrainViz;
            }

            onGameOver();

            if (!GameOver)
            {
                background.Update(gameTime);

                for (int p = 0; p < players.Count; p++)
                {
                    players[p].Update(gameTime);
                }

                for (int c = enemies.Count - 1; c >= 0; c--)
                {
                    enemies[c].Update(speed);
                }

                if (every_sec >= spawnIntervalFrames)
                {
                    every_sec = 0;
                    spawnIntervalFrames = NextSpawnIntervalFrames();
                    Spawn_Enemy();
                }

                every_sec += 1;
                roundElapsedSeconds += gameTime.ElapsedGameTime.TotalSeconds;
                speed += (float)(settings.SpeedRampPerSecond * roundElapsedSeconds);
            }
        }

        void GameStart()
        {
            enemies.Clear();

            EvolvePopulation();

            generation++;
            every_sec = 0;
            spawnIntervalFrames = settings.EnemySpawnIntervalFrames;
            speed = speedStart;
            roundElapsedSeconds = 0d;
            GameOver = false;
        }

        void EvolvePopulation()
        {
            var brains = players.Select(p => p.Brain).ToList();
            var fitnessScores = players.Select(p => p.Fitness).ToList();

            var rankedPlayers = players.OrderByDescending(p => p.Fitness).ToList();
            int eliteToExport = Math.Min(settings.MaxEliteGenomesToSave, rankedPlayers.Count);
            var ranked = rankedPlayers
                .Take(eliteToExport)
                .Select(p => (Genome: p.Brain.ExportGenome(), Fitness: p.Fitness))
                .ToList();

            var nextGenerationBrains = evolutionStrategy.NextGeneration(brains, fitnessScores).ToList();

            float bestThisRound = rankedPlayers.Count > 0 ? rankedPlayers[0].Fitness : 0f;
            if (bestGenomeEver != null && bestThisRound < bestFitnessEver && nextGenerationBrains.Count > 0)
            {
                nextGenerationBrains[nextGenerationBrains.Count - 1] = NeuralNetwork.FromGenome(bestGenomeEver);
            }

            for (int i = 0; i < players.Count; i++)
            {
                players[i].SetBrain(nextGenerationBrains[i]);
                players[i].Reset();
            }

            var saveData = DinoEvolutionStore.BuildSaveData(ranked, generation + 1, bestFitnessEver, bestGenomeEver, settings.MaxEliteGenomesToSave);
            bestFitnessEver = saveData.BestFitnessEver;
            bestGenomeEver = saveData.BestGenomeEver;
            DinoEvolutionStore.Save(saveData);
        }

        void onGameOver()
        {
            alive = players.Where(x => !x.dead).Count();

            if (alive == 0)
            {
                GameOver = true;
            }
        }

        protected override void Draw(GameTime gameTime)
        {
            bool menuBackdrop = state == GameState.MainMenu || state == GameState.Options || state == GameState.ConfirmNewTraining;
            GraphicsDevice.Clear(menuBackdrop ? Color.Black : Color.White);
            _spriteBatch.Begin();

            switch (state)
            {
                case GameState.MainMenu:
                    DrawMainMenuScreen();
                    break;

                case GameState.Options:
                    DrawOptionsScreen();
                    break;

                case GameState.ConfirmNewTraining:
                    DrawConfirmNewTrainingScreen();
                    break;

                case GameState.Paused:
                    DrawGameplay();
                    DrawPauseScreen();
                    break;

                case GameState.Playing:
                    DrawGameplay();
                    break;
            }

            _spriteBatch.End();
            base.Draw(gameTime);
        }

        void DrawGameplay()
        {
            background.Draw(_spriteBatch);

            for (int p = 0; p < players.Count; p++)
            {
                players[p].Draw(_spriteBatch);
            }

            for (int c = 0; c < enemies.Count; c++)
            {
                enemies[c].Draw(_spriteBatch);
            }

            DrawDebug();
            DrawBrainVisualizer();

            if (GameOver)
            {
                DrawGameOverScreen();
            }
        }

        float CenterX(string text, float scale)
        {
            return (_graphics.PreferredBackBufferWidth - font.MeasureString(text).X * scale) / 2f;
        }

        void DrawMainMenuScreen()
        {
            int screenW = _graphics.PreferredBackBufferWidth;
            int screenH = _graphics.PreferredBackBufferHeight;

            string title = "NEURAVAL DINO";
            string subtitle = "Entrenamiento evolutivo NEAT";
            float titleScale = 2.4f;
            float subtitleScale = 1f;

            Vector2 titlePos = new Vector2(CenterX(title, titleScale), screenH * 0.09f);
            Vector2 subtitlePos = new Vector2(CenterX(subtitle, subtitleScale), titlePos.Y + font.MeasureString(title).Y * titleScale + 8);

            _spriteBatch.DrawString(font, title, titlePos, Color.Gold, 0f, Vector2.Zero, titleScale, SpriteEffects.None, 0f);
            _spriteBatch.DrawString(font, subtitle, subtitlePos, Color.White, 0f, Vector2.Zero, subtitleScale, SpriteEffects.None, 0f);

            Vector2 menuOrigin = new Vector2(screenW / 2f - 150, screenH * 0.32f);
            mainMenu.Draw(_spriteBatch, font, menuOrigin, 38f, 1.05f);

            DrawTrainingSummaryPanel(screenH * 0.61f);

            string hint = "Flechas para moverte  -  Enter para elegir";
            Vector2 hintPos = new Vector2(CenterX(hint, 0.7f), screenH - 34);
            _spriteBatch.DrawString(font, hint, hintPos, new Color(200, 200, 200), 0f, Vector2.Zero, 0.7f, SpriteEffects.None, 0f);
        }

        string TrainingSummaryHeader()
        {
            if (players.Count > 0)
            {
                return "ENTRENAMIENTO EN CURSO";
            }

            return pendingSaveData != null ? "ENTRENAMIENTO GUARDADO" : "SIN ENTRENAMIENTO";
        }

        string[] TrainingSummaryLines()
        {
            if (players.Count > 0)
            {
                return new[]
                {
                    $"Generacion: {generation}",
                    $"Vivos: {alive} / {players.Count}",
                    $"Mejor fitness historico: {bestFitnessEver:0}"
                };
            }

            if (pendingSaveData != null)
            {
                int elites = pendingSaveData.EliteGenomes == null ? 0 : pendingSaveData.EliteGenomes.Count;

                return new[]
                {
                    $"Generacion: {pendingSaveData.Generation}",
                    $"Mejor fitness historico: {pendingSaveData.BestFitnessEver:0}",
                    $"Elites guardados: {elites}",
                    $"Mejor genoma guardado: {(pendingSaveData.BestGenomeEver == null ? "no" : "si")}"
                };
            }

            return new[] { "No hay ningun entrenamiento guardado." };
        }

        void DrawTrainingSummaryPanel(float topY)
        {
            int screenW = _graphics.PreferredBackBufferWidth;

            string header = TrainingSummaryHeader();
            string[] lines = TrainingSummaryLines();

            const float headerScale = 0.95f;
            const float lineScale = 0.85f;
            float lineHeight = font.MeasureString("Ag").Y * lineScale + 6f;
            float headerWidth = font.MeasureString(header).X * headerScale;
            float headerHeight = font.MeasureString(header).Y * headerScale;

            float contentWidth = headerWidth;

            foreach (string line in lines)
            {
                contentWidth = Math.Max(contentWidth, font.MeasureString(line).X * lineScale);
            }

            int panelWidth = (int)contentWidth + 60;
            int panelHeight = (int)(headerHeight + lines.Length * lineHeight) + 42;

            var panelRect = new Rectangle((screenW - panelWidth) / 2, (int)topY, panelWidth, panelHeight);

            DrawManager.DrawLine(_spriteBatch, panelRect, new Color(18, 18, 18, 220));
            DrawManager.DrawRectOutline(_spriteBatch, panelRect, new Color(90, 90, 90));

            Vector2 headerPos = new Vector2(panelRect.X + (panelRect.Width - headerWidth) / 2f, panelRect.Y + 14);
            _spriteBatch.DrawString(font, header, headerPos, Color.Gold, 0f, Vector2.Zero, headerScale, SpriteEffects.None, 0f);

            for (int i = 0; i < lines.Length; i++)
            {
                Vector2 linePos = new Vector2(panelRect.X + 30, panelRect.Y + 14 + headerHeight + 12 + i * lineHeight);
                _spriteBatch.DrawString(font, lines[i], linePos, Color.White, 0f, Vector2.Zero, lineScale, SpriteEffects.None, 0f);
            }
        }

        void DrawConfirmNewTrainingScreen()
        {
            int screenW = _graphics.PreferredBackBufferWidth;
            int screenH = _graphics.PreferredBackBufferHeight;

            string title = "NUEVO ENTRENAMIENTO";
            Vector2 titlePos = new Vector2(CenterX(title, 1.7f), screenH * 0.14f);
            _spriteBatch.DrawString(font, title, titlePos, Color.Gold, 0f, Vector2.Zero, 1.7f, SpriteEffects.None, 0f);

            string warning = "Se borrara el entrenamiento guardado y no se puede deshacer.";
            Vector2 warningPos = new Vector2(CenterX(warning, 0.9f), screenH * 0.28f);
            _spriteBatch.DrawString(font, warning, warningPos, new Color(255, 120, 120), 0f, Vector2.Zero, 0.9f, SpriteEffects.None, 0f);

            DrawTrainingSummaryPanel(screenH * 0.4f);

            Vector2 menuOrigin = new Vector2(screenW / 2f - 230, screenH * 0.7f);
            confirmNewTrainingMenu.Draw(_spriteBatch, font, menuOrigin, 38f, 1f);

            string hint = "Flechas para moverte  -  Enter para elegir  -  Esc para cancelar";
            Vector2 hintPos = new Vector2(CenterX(hint, 0.7f), screenH - 34);
            _spriteBatch.DrawString(font, hint, hintPos, new Color(200, 200, 200), 0f, Vector2.Zero, 0.7f, SpriteEffects.None, 0f);
        }

        void DrawOptionsScreen()
        {
            int screenW = _graphics.PreferredBackBufferWidth;
            int screenH = _graphics.PreferredBackBufferHeight;

            string title = "OPCIONES";
            Vector2 titlePos = new Vector2(CenterX(title, 1.8f), screenH * 0.2f);
            _spriteBatch.DrawString(font, title, titlePos, Color.Gold, 0f, Vector2.Zero, 1.8f, SpriteEffects.None, 0f);

            Vector2 menuOrigin = new Vector2(screenW / 2f - 180, screenH * 0.42f);
            optionsMenu.Draw(_spriteBatch, font, menuOrigin, 40f, 1f);

            string hint = "Enter / flechas para cambiar  -  Esc para volver";
            Vector2 hintPos = new Vector2(CenterX(hint, 0.7f), screenH - 40);
            _spriteBatch.DrawString(font, hint, hintPos, new Color(200, 200, 200), 0f, Vector2.Zero, 0.7f, SpriteEffects.None, 0f);
        }

        void DrawPauseScreen()
        {
            int screenW = _graphics.PreferredBackBufferWidth;
            int screenH = _graphics.PreferredBackBufferHeight;

            DrawManager.DrawLine(_spriteBatch, new Rectangle(0, 0, screenW, screenH), new Color(0, 0, 0, 170));

            string title = "PAUSA";
            Vector2 titlePos = new Vector2(CenterX(title, 2f), screenH * 0.25f);
            _spriteBatch.DrawString(font, title, titlePos, Color.Gold, 0f, Vector2.Zero, 2f, SpriteEffects.None, 0f);

            Vector2 menuOrigin = new Vector2(screenW / 2f - 100, screenH * 0.45f);
            pauseMenu.Draw(_spriteBatch, font, menuOrigin, 40f, 1.1f);
        }

        void Spawn_Enemy()
        {
            BaseEnemy enemy = probabilidad.GenerarConProbabilidad(settings.BirdSpawnProbabilityPercent)
                ? new Bird()
                : new Cactus();

            enemy.SetSpawnX(NextSpawnX());
            enemies.Add(enemy);
        }

        int NextSpawnIntervalFrames()
        {
            int jitter = Math.Max(0, settings.EnemySpawnIntervalJitterFrames);
            return settings.EnemySpawnIntervalFrames + Random.Shared.Next(0, jitter + 1);
        }

        int NextSpawnX()
        {
            int jitter = Math.Max(0, settings.EnemySpawnXJitter);
            return SpawnX - Random.Shared.Next(0, jitter + 1);
        }

        void DrawGameOverScreen()
        {
            int screenW = _graphics.PreferredBackBufferWidth;
            int screenH = _graphics.PreferredBackBufferHeight;

            string title = "GAME OVER";
            string genText = $"Generacion {generation}";
            string hint = "Presiona ENTER, ESPACIO o ESC para continuar";

            Vector2 titleSize = font.MeasureString(title) * 2f;
            Vector2 genSize = font.MeasureString(genText) * 1.2f;
            Vector2 hintSize = font.MeasureString(hint);

            int panelWidth = (int)Math.Max(titleSize.X, Math.Max(genSize.X, hintSize.X)) + 80;
            int panelHeight = 180;
            var panelRect = new Rectangle(
                (screenW - panelWidth) / 2,
                (screenH - panelHeight) / 2,
                panelWidth,
                panelHeight);
            DrawManager.DrawLine(_spriteBatch, panelRect, new Color(0, 0, 0, 160));

            Vector2 titlePos = new Vector2((screenW - titleSize.X) / 2f, panelRect.Y + 20);
            Vector2 genPos = new Vector2((screenW - genSize.X) / 2f, titlePos.Y + titleSize.Y + 15);
            Vector2 hintPos = new Vector2((screenW - hintSize.X) / 2f, genPos.Y + genSize.Y + 20);

            _spriteBatch.DrawString(font, title, titlePos, Color.Red, 0f, Vector2.Zero, 2f, SpriteEffects.None, 0f);
            _spriteBatch.DrawString(font, genText, genPos, Color.White, 0f, Vector2.Zero, 1.2f, SpriteEffects.None, 0f);
            _spriteBatch.DrawString(font, hint, hintPos, Color.Yellow);
        }

        void DrawDebug()
        {
            for (int i = 0; i < players.Count; i++)
            {
                if (!players[i].dead)
                {
                    playerTarget = players[i];
                    break;
                }
            }

            _spriteBatch.DrawString(font, $"Generacion: {generation}", new Vector2(10, 20), Color.Black);
            _spriteBatch.DrawString(font, $"Vivos: {alive}", new Vector2(10, 40), Color.Black);
            _spriteBatch.DrawString(font, $"Mejor fitness historico: {bestFitnessEver:0}", new Vector2(10, 60), Color.Black);
            _spriteBatch.DrawString(font, "[F2] Mostrar/ocultar red neuronal  -  [Esc] Pausa", new Vector2(10, 80), Color.Black, 0f, Vector2.Zero, 0.8f, SpriteEffects.None, 0f);
        }

        void DrawBrainVisualizer()
        {
            if (!showBrainViz || GameOver || playerTarget == null || playerTarget.dead)
            {
                return;
            }

            var snapshot = playerTarget.CaptureBrainSnapshot();

            int panelWidth = 360;
            int panelHeight = 420;
            var panelRect = new Rectangle(
                _graphics.PreferredBackBufferWidth - panelWidth - 20,
                70,
                panelWidth,
                panelHeight);

            NeuralNetworkVisualizer.Draw(_spriteBatch, font, snapshot, panelRect);
        }
    }
}
