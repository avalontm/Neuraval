namespace Neuraval.Abstractions
{
    public interface ITrainer<TModel, TDataset> where TModel : ITrainableModel
    {
        void Train(TModel model, TDataset dataset);
    }
}
