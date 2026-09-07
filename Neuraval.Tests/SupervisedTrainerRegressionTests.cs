using System.Collections.Generic;
using Neuraval.Abstractions;
using Neuraval.Core.Models;
using Neuraval.Core.Services;
using Neuraval.Core.Utils;
using Xunit;

namespace Neuraval.Tests
{
    public class SupervisedTrainerRegressionTests
    {
        private const int VocabSize = 10;
        private const int EmbeddingDim = 8;
        private const int NumLayers = 1;
        private const int NumHeads = 2;
        private const int FeedforwardDim = 16;
        private const int MaxSequenceLength = 8;

        [Fact]
        public void TrainViaITrainer_MatchesDirectTrainCausalWithValidation()
        {
            Matematicas.SetNumThreads(1);

            var modelA = CreateModel();
            var modelB = CreateModel();

            var trainerA = new SupervisedTrainer(modelA, learningRate: 0.01f, shuffleSeed: 7, padToken: 0);
            var trainerB = new SupervisedTrainer(modelB, learningRate: 0.01f, shuffleSeed: 7, padToken: 0);

            var trainingExamples = CreateExamples();
            var validationExamples = CreateExamples();

            trainerA.TrainCausalWithValidation(
                trainingExamples,
                validationExamples,
                epochs: 2,
                batchSize: 2,
                patience: 20);

            ITrainer<TransformerModel, CausalTrainingDataset> universalTrainer = trainerB;
            universalTrainer.Train(modelB, new CausalTrainingDataset
            {
                TrainingExamples = trainingExamples,
                ValidationExamples = validationExamples,
                Epochs = 2,
                BatchSize = 2,
                Patience = 20
            });

            Assert.Equal(trainerA.GetTrainingLosses(), trainerB.GetTrainingLosses());
            Assert.Equal(trainerA.GetValidationLosses(), trainerB.GetValidationLosses());
            Assert.Equal(trainerA.BestValidationLoss, trainerB.BestValidationLoss);

            var sampleTokens = new[] { 1, 2, 3, 4 };
            Assert.Equal(modelA.Predict(sampleTokens), modelB.Predict(sampleTokens));
        }

        private static TransformerModel CreateModel()
        {
            return new TransformerModel(
                vocabSize: VocabSize,
                embeddingDim: EmbeddingDim,
                numLayers: NumLayers,
                numHeads: NumHeads,
                feedforwardDim: FeedforwardDim,
                maxSequenceLength: MaxSequenceLength,
                dropout: 0f,
                seed: 123);
        }

        private static List<CausalExample> CreateExamples()
        {
            return new List<CausalExample>
            {
                new CausalExample { Tokens = new[] { 1, 2, 3, 4, 5, 0 }, ResponseStartIndex = 3 },
                new CausalExample { Tokens = new[] { 2, 3, 4, 5, 6, 0 }, ResponseStartIndex = 3 },
                new CausalExample { Tokens = new[] { 3, 4, 5, 6, 7, 0 }, ResponseStartIndex = 3 },
                new CausalExample { Tokens = new[] { 4, 5, 6, 7, 8, 0 }, ResponseStartIndex = 3 }
            };
        }
    }
}
