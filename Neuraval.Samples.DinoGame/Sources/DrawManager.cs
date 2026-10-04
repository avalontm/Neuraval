using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuraval.Samples.DinoGame.Sources
{
    public static class DrawManager
    {
        static Texture2D t;

        public static void Init(GraphicsDevice g)
        {
            t = new Texture2D(g, 1, 1);
            t.SetData<Color>(new Color[] { Color.White });
        }

        public static void DrawLine(SpriteBatch sb, Rectangle rec, Color color)
        {
            sb.Draw(t, rec, color);
        }

        public static void DrawLineSegment(SpriteBatch sb, Vector2 start, Vector2 end, Color color, float thickness = 1f)
        {
            Vector2 delta = end - start;
            float length = delta.Length();
            if (length < 0.001f)
            {
                return;
            }

            float angle = (float)Math.Atan2(delta.Y, delta.X);
            sb.Draw(t, start, null, color, angle, Vector2.Zero, new Vector2(length, thickness), SpriteEffects.None, 0f);
        }

        public static void DrawRectOutline(SpriteBatch sb, Rectangle rec, Color color)
        {
            DrawLine(sb, new Rectangle(rec.X, rec.Y, rec.Width, 1), color);
            DrawLine(sb, new Rectangle(rec.X, rec.Bottom - 1, rec.Width, 1), color);
            DrawLine(sb, new Rectangle(rec.X, rec.Y, 1, rec.Height), color);
            DrawLine(sb, new Rectangle(rec.Right - 1, rec.Y, 1, rec.Height), color);
        }
    }
}
