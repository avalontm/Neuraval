using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuraval.Samples.DinoGame.Sources
{
    
    public class BaseEnemy 
    {
        public int x { set; get; }
        public int y { set; get; }
        public int w { set; get; }
        public int h { set; get; }

        public int type { set; get; }

        public Rectangle Bounds { internal set; get; }

        public BaseEnemy()
        {

        }

        public virtual void Update(double speed)
        {
            if (x < -10)
            {
                MainGame.enemies.Remove(this);
            }
        }

        public virtual void Draw(SpriteBatch _spriteBatch)
        {

        }
    }
}
