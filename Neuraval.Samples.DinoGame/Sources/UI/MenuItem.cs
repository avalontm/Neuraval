using System;

namespace Neuraval.Samples.DinoGame.Sources.UI
{
    public class MenuItem
    {
        public Func<string> Label { get; }
        public Action Activate { get; }

        public MenuItem(string label, Action activate) : this(() => label, activate)
        {
        }

        public MenuItem(Func<string> label, Action activate)
        {
            Label = label;
            Activate = activate;
        }
    }
}
