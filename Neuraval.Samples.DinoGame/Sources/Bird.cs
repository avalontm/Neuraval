using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;

namespace Neuraval.Samples.DinoGame.Sources
{
    public class Bird : BaseEnemy
    {
        int fotogramaActual;
        float tiempoTranscurrido;
        float tiempoCambioFotograma = 0.1f;
        int totalFotogramas = 2;

        public Bird()
        {
            x = MainGame.SpawnX;
            w = 84;
            h = 40;
            type = Random.Shared.Next(4);

            switch (type)
            {
                case 0:
                    y = 365;
                    break;
                case 1:
                    y = 440;
                    break;
                case 2:
                    y = 465;
                    break;
                case 3:
                    y = 345;
                    h = 145;
                    break;
            }

            Bounds = new Rectangle(x, y, w, h);
        }

        void Animation()
        {
            tiempoTranscurrido += (float)MainGame.time.ElapsedGameTime.TotalSeconds;

            if (tiempoTranscurrido >= tiempoCambioFotograma)
            {
                fotogramaActual = (fotogramaActual + 1) % totalFotogramas;
                tiempoTranscurrido = 0;
            }
        }

        public override void Update(double speed)
        {
            base.Update(speed);

            x -= (int)speed;

            Animation();
        }

        public override void Draw(SpriteBatch _spriteBatch)
        {
            base.Draw(_spriteBatch);

            int offset = 5;
            Bounds = new Rectangle(x + (offset+5), y + offset, w - (offset * 2), h - (offset*2));
            Rectangle rec = new Rectangle(x, y, w, h);

            _spriteBatch.Draw(Animations.birds[fotogramaActual], rec, Color.White);

            if (MainGame.IsDebug)
            {
                onDebug(_spriteBatch);
            }
        }

        void onDebug(SpriteBatch _spriteBatch)
        {
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X, Bounds.Y, Bounds.Width, 1), Color.Red);
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X, Bounds.Y, 1, Bounds.Height), Color.Red);
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X + Bounds.Width, Bounds.Y, 1, Bounds.Height), Color.Red);
            DrawManager.DrawLine(_spriteBatch, new Rectangle(Bounds.X, Bounds.Y + Bounds.Height, Bounds.Width, 1), Color.Red);
        }
    }
}
