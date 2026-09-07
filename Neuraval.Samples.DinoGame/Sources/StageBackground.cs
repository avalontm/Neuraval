using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

namespace Neuraval.Samples.DinoGame.Sources
{
    
    public class StageBackground
    {
        public int x { set; get; }
        public int y { set; get; }
        public int w { set; get; }
        public int h { set; get; }

        Texture2D rect { set; get; }

        public StageBackground()
        {
            rect = new Texture2D(MainGame._graphics.GraphicsDevice, 1, 1);
            rect.SetData(new[] { Color.White });

            x = 0;
            y = 535;
            w = 1280;
            h = 28;
        }


        public void Update(GameTime gameTime)
        {

          
        }

        public void Draw(SpriteBatch _spriteBatch)
        {
            Rectangle coords = new Rectangle((int)x, (int)(y-(h/2))-8, w, h);

            _spriteBatch.Draw(Animations.stage[0], coords, Color.White);
        }
    }
}
