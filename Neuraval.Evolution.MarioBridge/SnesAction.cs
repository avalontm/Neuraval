using System;
using System.Linq;

namespace Neuraval.Evolution.MarioBridge
{
    public readonly struct SnesAction
    {
        public static readonly SnesAction None = new SnesAction(SnesButton.None);

        public SnesButton Buttons { get; }

        public SnesAction(SnesButton buttons)
        {
            Buttons = buttons;
        }

        public bool IsPressed(SnesButton button)
        {
            return (Buttons & button) == button;
        }

        public string ToWireFormat()
        {
            if (Buttons == SnesButton.None)
            {
                return "None";
            }

            var buttons = Buttons;
            var pressed = Enum.GetValues<SnesButton>()
                .Where(button => button != SnesButton.None && (buttons & button) == button)
                .Select(button => button.ToString());

            return string.Join(",", pressed);
        }
    }
}
