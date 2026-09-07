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
    }
}
