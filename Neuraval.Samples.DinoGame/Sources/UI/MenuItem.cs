using System;

namespace Neuraval.Samples.DinoGame.Sources.UI
{
    public class MenuItem
    {
        public Func<string> Label { get; }
        public Action Activate { get; }
        public Action<int> Adjust { get; }

        public MenuItem(string label, Action activate) : this(() => label, activate, null)
        {
        }

        public MenuItem(Func<string> label, Action activate) : this(label, activate, null)
        {
        }

        public MenuItem(Func<string> label, Action activate, Action<int> adjust)
        {
            Label = label;
            Activate = activate;
            Adjust = adjust;
        }
    }
}
