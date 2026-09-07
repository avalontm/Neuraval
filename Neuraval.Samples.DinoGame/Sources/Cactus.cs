using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;

namespace Neuraval.Samples.DinoGame.Sources
{
    public class Cactus : BaseEnemy
    {
        public Cactus()
        {
            Random rnd = new Random();

            x = 1350;
            type = (int)rnd.Next(6);

            if (type < 3)
            {
                h = 66;
                y = 470;
            }
            else
            {
                h = 96;
                y = 440;
            }

            switch (type)
            {
                case 0:
                    w = 30;
                    break;
                case 1:
                    w = 64;
                    break;
                case 2:
                    w = 98;
                    break;
                case 3:
                    w = 46;
                    break;
                case 4:
                    w = 96;
                    break;
                case 5:
                    w = 146;
                    break;
            }

            Bounds = new Rectangle(x, y, w, h);
        }


        public override void Update(double speed)
        {
            base.Update(speed);
            x -= (int)speed;

        }

        public override void Draw(SpriteBatch _spriteBatch)
        {
            base.Draw(_spriteBatch);

            int offset = 10;
            Bounds = new Rectangle(x + offset, y + offset, w - (offset * 2), h - offset);
            Rectangle rec = new Rectangle(x, y, w, h);
            _spriteBatch.Draw(Animations.cactus[type], rec, Color.White);

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
