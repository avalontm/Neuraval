using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuraval.Samples.DinoGame.Sources
{
    public static class Animations
    {
        public static Texture2D[] stage { private set; get; }
        public static Texture2D[] dino { private set; get; }
        public static Texture2D[] cactus { private set; get; }

        public static Texture2D[] birds { private set; get; }

        public static void Load(ContentManager Content)
        {
            stage = new Texture2D[1];
            dino = new Texture2D[3];
            cactus = new Texture2D[6];
            birds = new Texture2D[2];

            stage[0] = Content.Load<Texture2D>("ground");

            dino[0] = Content.Load<Texture2D>("dinosaur");
            birds[0] = Content.Load<Texture2D>("bird_1");
            birds[1] = Content.Load<Texture2D>("bird_2");

            for (int i = 0; i < 6; i++)
            {
                cactus[i] = Content.Load<Texture2D>("cactus");
            }
        }
    }
}
