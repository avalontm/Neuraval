using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;

namespace Neuraval.Samples.DinoGame.Sources
{
    
    public class Bird : BaseEnemy
    {
        //Animacion
        int fotogramaActual;
        float tiempoTranscurrido;
        float tiempoCambioFotograma = 0.1f; // Cambia el fotograma cada 0.1 segundos
        int totalFotogramas = 2;

        public Bird()
        {
            Random rnd = new Random();

            x = 1350;
            w = 84;
            h = 40;
            // type 3 es un pajaro especial (ver mas abajo) que sobreescribe
            // "h" con un valor mucho mayor, asi que se sortea aparte para
            // dejar claro que no es "uno mas" del mismo tamaño.
            type = (int)rnd.Next(4);

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
                    // Pajaro "bajo" (obliga a agacharse): a diferencia de
                    // los otros tres tipos, su rango vertical de colision
                    // cubre TODO el arco de salto del dino (desde el pico
                    // del salto hasta el suelo), asi que saltar nunca lo
                    // esquiva, sin importar el timing. Agacharse si lo
                    // esquiva porque baja el perfil de colision del dino
                    // por debajo de la parte inferior de este pajaro. Este
                    // es el obstaculo que le da a la evolucion una razon
                    // real para aprender a usar "agacharse".
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
                // Cambiar al siguiente fotograma de la animación
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
