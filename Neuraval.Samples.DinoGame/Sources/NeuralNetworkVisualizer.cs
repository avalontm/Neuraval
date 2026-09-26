using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using System;

namespace Neuraval.Samples.DinoGame.Sources
{
    /// <summary>
    /// Dibuja, como panel de HUD, la red neuronal (entrada -&gt; capa
    /// oculta -&gt; salida) de un <see cref="NetworkActivationSnapshot"/>
    /// en tiempo real: un nodo por neurona (coloreado segun su activacion)
    /// y una linea por conexion (verde = peso excitatorio, rojo = peso
    /// inhibitorio; el grosor/opacidad refleja la magnitud del peso).
    ///
    /// Es puramente de lectura/dibujo: no conoce nada del entrenamiento ni
    /// del algoritmo genetico, solo consume el snapshot que ya calculo
    /// <see cref="NeuralNetwork.GetActivationSnapshot"/>.
    /// </summary>
    public static class NeuralNetworkVisualizer
    {
        static readonly string[] InputLabels =
        {
            "Dist.", "ObsX", "ObsY", "Ancho", "Alto", "DinoY", "Vel."
        };

        static readonly string[] OutputLabels = { "Saltar", "Agachar" };

        const int NodeSize = 16;

        public static void Draw(SpriteBatch spriteBatch, SpriteFont font, NetworkActivationSnapshot snapshot, Rectangle panel)
        {
            if (snapshot == null)
            {
                return;
            }

            // Fondo semitransparente para que las lineas/nodos resalten
            // sobre el juego sin tapar del todo lo que hay detras.
            DrawManager.DrawLine(spriteBatch, panel, new Color(15, 15, 20, 165));
            DrawManager.DrawRectOutline(spriteBatch, panel, new Color(255, 255, 255, 90));

            int inputCount = snapshot.Inputs.Length;
            int hiddenCount = snapshot.Hidden.Length;
            int outputCount = snapshot.Outputs.Length;

            Vector2[] inputPos = LayoutColumn(panel, 0.14f, inputCount);
            Vector2[] hiddenPos = LayoutColumn(panel, 0.52f, hiddenCount);
            Vector2[] outputPos = LayoutColumn(panel, 0.88f, outputCount);

            // Conexiones primero, para que los nodos se dibujen encima.
            for (int i = 0; i < inputCount; i++)
            {
                for (int j = 0; j < hiddenCount; j++)
                {
                    DrawConnection(spriteBatch, inputPos[i], hiddenPos[j], snapshot.InputToHiddenWeights[i, j]);
                }
            }

            for (int j = 0; j < hiddenCount; j++)
            {
                for (int k = 0; k < outputCount; k++)
                {
                    DrawConnection(spriteBatch, hiddenPos[j], outputPos[k], snapshot.HiddenToOutputWeights[j, k]);
                }
            }

            for (int i = 0; i < inputCount; i++)
            {
                string label = i < InputLabels.Length ? InputLabels[i] : "";
                DrawNode(spriteBatch, font, inputPos[i], snapshot.Inputs[i], label, false, true);
            }

            for (int j = 0; j < hiddenCount; j++)
            {
                DrawNode(spriteBatch, font, hiddenPos[j], snapshot.Hidden[j], null, false, false);
            }

            // La decision real del Dino (ver Dino.onIA): solo cuenta si al
            // menos una señal supera 0; si no, sigue corriendo y ninguna
            // salida se resalta como "ganadora".
            bool anyActive = snapshot.Outputs[0] > 0f || snapshot.Outputs[1] > 0f;
            int winner = snapshot.Outputs[0] >= snapshot.Outputs[1] ? 0 : 1;

            for (int k = 0; k < outputCount; k++)
            {
                string label = k < OutputLabels.Length ? OutputLabels[k] : "";
                DrawNode(spriteBatch, font, outputPos[k], snapshot.Outputs[k], label, anyActive && k == winner, true);
            }

            spriteBatch.DrawString(font, "Como piensa el Dino", new Vector2(panel.X + 14, panel.Y + 8), Color.White, 0f, Vector2.Zero, 0.9f, SpriteEffects.None, 0f);
        }

        static Vector2[] LayoutColumn(Rectangle panel, float xFraction, int count)
        {
            var positions = new Vector2[count];
            if (count == 0)
            {
                return positions;
            }

            float x = panel.X + panel.Width * xFraction;
            float top = panel.Y + 42;
            float bottom = panel.Y + panel.Height - 24;
            float step = count > 1 ? (bottom - top) / (count - 1) : 0f;

            for (int i = 0; i < count; i++)
            {
                float y = count > 1 ? top + step * i : (top + bottom) / 2f;
                positions[i] = new Vector2(x, y);
            }

            return positions;
        }

        static void DrawConnection(SpriteBatch spriteBatch, Vector2 from, Vector2 to, float weight)
        {
            // Normaliza la magnitud del peso a [0,1] (1.5 cubre con margen
            // el rango tipico de pesos tras Xavier/Glorot + mutacion) para
            // que la opacidad/grosor sean comparables entre conexiones.
            float magnitude = MathHelper.Clamp(Math.Abs(weight) / 1.5f, 0f, 1f);

            byte alpha = (byte)MathHelper.Clamp(25 + magnitude * 210f, 0f, 255f);
            Color color = weight >= 0f
                ? new Color((byte)50, (byte)200, (byte)100, alpha)   // excitatorio
                : new Color((byte)220, (byte)70, (byte)70, alpha);   // inhibitorio

            float thickness = 1f + magnitude * 2.5f;
            DrawManager.DrawLineSegment(spriteBatch, from, to, color, thickness);
        }

        static void DrawNode(SpriteBatch spriteBatch, SpriteFont font, Vector2 center, float activation, string label, bool highlight, bool labelAbove)
        {
            // Intensidad del color segun que tan lejos de 0 esta la
            // activacion; 2 cubre con margen el rango tipico de una entrada
            // escalada o una salida de la ultima capa (activacion lineal).
            float intensity = MathHelper.Clamp(Math.Abs(activation) / 2f, 0f, 1f);

            Color fill;
            if (highlight)
            {
                fill = Color.Gold;
            }
            else if (activation >= 0f)
            {
                fill = Color.Lerp(new Color(55, 55, 65), new Color(80, 220, 120), intensity);
            }
            else
            {
                fill = Color.Lerp(new Color(55, 55, 65), new Color(220, 80, 80), intensity);
            }

            var rect = new Rectangle((int)center.X - NodeSize / 2, (int)center.Y - NodeSize / 2, NodeSize, NodeSize);
            DrawManager.DrawLine(spriteBatch, rect, fill);
            DrawManager.DrawRectOutline(spriteBatch, rect, Color.White);

            if (!string.IsNullOrEmpty(label))
            {
                Vector2 size = font.MeasureString(label) * 0.7f;
                float labelY = labelAbove ? center.Y - NodeSize - size.Y - 2 : center.Y + NodeSize / 2 + 2;
                Vector2 textPos = new Vector2(center.X - size.X / 2f, labelY);
                spriteBatch.DrawString(font, label, textPos, Color.White, 0f, Vector2.Zero, 0.7f, SpriteEffects.None, 0f);
            }
        }
    }
}
