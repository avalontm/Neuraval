using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Collections.Generic;

namespace Neuraval.Samples.DinoGame.Sources.UI
{
    public class Menu
    {
        readonly List<MenuItem> items;
        int selectedIndex;

        public Menu(IEnumerable<MenuItem> items)
        {
            this.items = new List<MenuItem>(items);
        }

        public void Reset()
        {
            selectedIndex = 0;
        }

        public void Update()
        {
            if (items.Count == 0)
            {
                return;
            }

            if (InputManager.IsKeyPressed(Keys.Down, true) || InputManager.IsKeyPressed(Keys.S, true))
            {
                selectedIndex = (selectedIndex + 1) % items.Count;
            }

            if (InputManager.IsKeyPressed(Keys.Up, true) || InputManager.IsKeyPressed(Keys.W, true))
            {
                selectedIndex = (selectedIndex - 1 + items.Count) % items.Count;
            }

            MenuItem current = items[selectedIndex];

            if (current.Adjust != null)
            {
                if (InputManager.IsKeyPressed(Keys.Right, true) || InputManager.IsKeyPressed(Keys.D, true))
                {
                    current.Adjust(1);
                }

                if (InputManager.IsKeyPressed(Keys.Left, true) || InputManager.IsKeyPressed(Keys.A, true))
                {
                    current.Adjust(-1);
                }
            }

            if (InputManager.IsKeyPressed(Keys.Enter, true) || InputManager.IsKeyPressed(Keys.Space, true))
            {
                current.Activate?.Invoke();
            }
        }

        public void Draw(SpriteBatch spriteBatch, SpriteFont font, Vector2 origin, float spacing, float scale = 1f)
        {
            for (int i = 0; i < items.Count; i++)
            {
                bool selected = i == selectedIndex;
                Color color = selected ? Color.Gold : Color.White;
                string prefix = selected ? "> " : "  ";
                Vector2 position = origin + new Vector2(0, i * spacing);

                spriteBatch.DrawString(font, prefix + items[i].Label(), position, color, 0f, Vector2.Zero, scale, SpriteEffects.None, 0f);
            }
        }
    }
}
