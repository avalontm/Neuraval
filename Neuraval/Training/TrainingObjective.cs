namespace Neuraval.Core.Training
{
    public abstract class TrainingObjective
    {
        public abstract string Name { get; }

        public abstract LabelMask BuildLabelMask(TrainingSequence sequence);
    }
}
