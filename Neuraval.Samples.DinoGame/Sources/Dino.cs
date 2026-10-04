using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Linq;

namespace Neuraval.Samples.DinoGame.Sources
{
    public class Dino
    {
        public int x { set; get; }
        public int y { set; get; }
        public int w { set; get; }
        public int h { set; get; }
        public bool jumping { set; get; }
        public bool crounching { set; get; }
        public float jump_stage { set; get; }
        public bool dead { set; get; }

        public Rectangle Bounds { private set; get; }

        NeuralNetwork neuralNetwork;

        public NeuralNetwork Brain => neuralNetwork;

        public NetworkActivationSnapshot CaptureBrainSnapshot()
        {
            return neuralNetwork.GetActivationSnapshot();
        }

        public float Fitness { private set; get; }

        int fotogramaActual;
        float tiempoTranscurrido;
        float tiempoCambioFotograma = 0.1f;

        public Dino()
        {
            neuralNetwork = new NeuralNetwork();

            Reset();
        }

        public Dino(NeuralNetwork brain)
        {
            neuralNetwork = brain;

            Reset();
        }

        public void SetBrain(NeuralNetwork brain)
        {
            neuralNetwork = brain;
        }

        public void Reset()
        {
            x = MainGame.DinoStartX + Random.Shared.Next(-MainGame.DinoStartXJitter, MainGame.DinoStartXJitter);
            y = 450;
            w = 80;
            h = 86;

            jumping = false;
            crounching = false;

            Bounds = new Rectangle(x, y, w, h);
            dead = false;
            Fitness = 0f;
        }

        public void Update(GameTime gameTime)
        {
            if (!dead)
            {
                Fitness += MainGame.speed * (float)gameTime.ElapsedGameTime.TotalSeconds;

                if (jumping)
                {
                    y = 448 - (int)f(jump_stage);
                    jump_stage += 0.03f;
                    if (jump_stage > 1)
                    {
                        jumping = false;
                        crounching = false;
                        jump_stage = 0;
                        y = 450;
                    }
                }
                else if (crounching)
                {
                    crounching = false;
                    y -= 34;
                    w = 80;
                    h = 86;
                }
                else
                {
                    jumping = false;
                    crounching = false;
                    jump_stage = 0;
                    y = 450;
                }

                onIA();

                onCollition();
            }
        }

        public void onIA()
        {
            float[] input = GetGameStatus();

            float[] output = neuralNetwork.Predict(input);

            float jumpSignal = output[0];
            float duckSignal = output[1];

            if (jumpSignal <= 0f && duckSignal <= 0f)
            {
                return;
            }

            if (jumpSignal >= duckSignal)
            {
                onJump();
            }
            else
            {
                onDuck();
            }
        }

        public int CalculateDistanceToObstacle()
        {
            int distance = 0;

            if (MainGame.enemies.Count > 0)
            {
                distance = MainGame.enemies[0].Bounds.X - Bounds.X;
                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        distance = MainGame.enemies[1].Bounds.X - Bounds.X;
                    }
                }
            }

            return distance;
        }

        public int CalculateObstaclePositionX()
        {
            int x = 0;

            if (MainGame.enemies.Count > 0)
            {
                int distance = CalculateDistanceToObstacle();

                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        x = MainGame.enemies[1].Bounds.X;
                    }
                }
                else
                {
                    x = MainGame.enemies[0].Bounds.X;
                }
            }

            return x;
        }

        public int CalculateObstaclePositionY()
        {
            int y = 0;

            if (MainGame.enemies.Count > 0)
            {
                int distance = CalculateDistanceToObstacle();

                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        y = MainGame.enemies[1].Bounds.Y;
                    }
                }
                else
                {
                    y = MainGame.enemies[0].Bounds.Y;
                }
            }

            return y;
        }

        public int CalculateObstacleWidth()
        {
            int w = 0;

            if (MainGame.enemies.Count > 0)
            {
                int distance = CalculateDistanceToObstacle();

                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        w = MainGame.enemies[1].Bounds.Width;
                    }
                }
                else
                {
                    w = MainGame.enemies[0].Bounds.Width;
                }
            }

            return w;
        }

        public int CalculateObstacleHeight()
        {
            int h = 0;

            if (MainGame.enemies.Count > 0)
            {
                int distance = CalculateDistanceToObstacle();

                if (distance < 0)
                {
                    if (MainGame.enemies.Count > 1)
                    {
                        h = MainGame.enemies[1].Bounds.Height;
                    }
                }
                else
                {
                    h = MainGame.enemies[0].Bounds.Height;
                }
            }

            return h;
        }

        float[] GetGameStatus()
        {
            float[] features = new float[7];
            features[0] = CalculateDistanceToObstacle();
            features[1] = CalculateObstaclePositionX();
            features[2] = CalculateObstaclePositionY();
            features[3] = CalculateObstacleWidth();
            features[4] = CalculateObstacleHeight();
            features[5] = Bounds.Y;
            features[6] = MainGame.speed;
            return features;
        }

        void onCollition()
        {
            for (int c = 0; c < MainGame.enemies.Count; c++)
            {
                if (Collition(MainGame.enemies[c]))
                {
                    if (MainGame.players.Where(x => !x.dead).Count() == 1)
                    {
                        MainGame.LastPlayer = this;
                    }

                    dead = true;
                }
            }
        }

        void onDebug(SpriteBatch _spriteBatch)
        {
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X, Bounds.Y, Bounds.Width, 1), Color.Red);
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X, Bounds.Y, 1, Bounds.Height), Color.Red);
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X + Bounds.Width, Bounds.Y, 1, Bounds.Height), Color.Red);
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X, Bounds.Y + Bounds.Height, Bounds.Width, 1), Color.Red);
        }

        public void Draw(SpriteBatch _spriteBatch)
        {
            if (!dead)
            {
                int offset = 20;
                Bounds = new Rectangle(x + offset, y + offset, w - (offset*2), h - (offset+5));
                Rectangle rec = new Rectangle(x , y , w , h );
                _spriteBatch.Draw(Animations.dino[0], rec, Color.White);

                if (MainGame.IsDebug)
                {
                    onDebug(_spriteBatch);
                }
            }
        }

        float f(float x)
        {
            return (-4 * x * (x - 1) * 172);
        }

        bool Collition(BaseEnemy enemy)
        {
            return this.Bounds.Intersects(enemy.Bounds);
        }

        public void onJump()
        {
            if (!jumping)
            {
                jumping = true;
                crounching = false;
            }
        }

        public void onDuck()
        {
            if (!jumping)
            {
                jumping = false;
                crounching = true;
                y += 34;
                w = 110;
                h = 52;
            }
        }
    }
}
