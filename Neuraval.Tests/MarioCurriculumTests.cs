using Neuraval.Evolution.MarioBridge;
using Xunit;

namespace Neuraval.Tests
{
    public class MarioCurriculumTests
    {
        [Fact]
        public void RecordGeneration_DoesNotAdvance_WhileWindowIsBelowCapacity()
        {
            var curriculum = new MarioCurriculum(stageCount: 3, windowSize: 5, advanceThresholdPercent: 50f);

            for (var i = 0; i < 4; i++)
            {
                var advanced = curriculum.RecordGeneration(100f);
                Assert.False(advanced);
            }

            Assert.Equal(0, curriculum.StageIndex);
            Assert.Equal(4, curriculum.WindowSampleCount);
        }

        [Fact]
        public void RecordGeneration_Advances_OnceWindowAverageMeetsThreshold()
        {
            var curriculum = new MarioCurriculum(stageCount: 3, windowSize: 4, advanceThresholdPercent: 50f);

            Assert.False(curriculum.RecordGeneration(40f));
            Assert.False(curriculum.RecordGeneration(40f));
            Assert.False(curriculum.RecordGeneration(40f));
            var advanced = curriculum.RecordGeneration(80f);

            Assert.True(advanced);
            Assert.Equal(1, curriculum.StageIndex);
            Assert.Equal(0, curriculum.GenerationsAtStage);
            Assert.Equal(0, curriculum.WindowSampleCount);
        }

        [Fact]
        public void RecordGeneration_DoesNotAdvance_WhenWindowAverageBelowThreshold()
        {
            var curriculum = new MarioCurriculum(stageCount: 2, windowSize: 4, advanceThresholdPercent: 50f);

            curriculum.RecordGeneration(0f);
            curriculum.RecordGeneration(0f);
            curriculum.RecordGeneration(0f);
            var advanced = curriculum.RecordGeneration(0f);

            Assert.False(advanced);
            Assert.Equal(0, curriculum.StageIndex);
        }

        [Fact]
        public void RecordGeneration_SlidesWindow_DiscardingOldestSample()
        {
            // stageCount=1 aisla el comportamiento de la ventana movil en si:
            // con un unico tramo (el ultimo), RecordGeneration jamas avanza,
            // asi que un eventual avance de tramo por el 100f inicial no
            // puede "enmascarar" lo que este test quiere probar (que las
            // muestras viejas se descartan de la ventana).
            var curriculum = new MarioCurriculum(stageCount: 1, windowSize: 3, advanceThresholdPercent: 50f);

            curriculum.RecordGeneration(100f);
            curriculum.RecordGeneration(100f);
            curriculum.RecordGeneration(0f);
            curriculum.RecordGeneration(0f);
            var advanced = curriculum.RecordGeneration(0f);

            Assert.False(advanced);
            Assert.Equal(3, curriculum.WindowSampleCount);
            Assert.Equal(0f, curriculum.WindowAverageCompletion);
        }

        [Fact]
        public void RecordGeneration_NeverAdvancesPastFinalStage()
        {
            var curriculum = new MarioCurriculum(stageCount: 1, windowSize: 2, advanceThresholdPercent: 10f);

            Assert.True(curriculum.IsAtFinalStage);
            Assert.False(curriculum.RecordGeneration(100f));
            Assert.False(curriculum.RecordGeneration(100f));
            Assert.False(curriculum.RecordGeneration(100f));
            Assert.Equal(0, curriculum.StageIndex);
            Assert.True(curriculum.IsAtFinalStage);
        }

        [Fact]
        public void RecordGeneration_CanAdvanceThroughMultipleStagesSequentially()
        {
            var curriculum = new MarioCurriculum(stageCount: 3, windowSize: 2, advanceThresholdPercent: 50f);

            curriculum.RecordGeneration(100f);
            var firstAdvance = curriculum.RecordGeneration(100f);
            Assert.True(firstAdvance);
            Assert.Equal(1, curriculum.StageIndex);

            curriculum.RecordGeneration(100f);
            var secondAdvance = curriculum.RecordGeneration(100f);
            Assert.True(secondAdvance);
            Assert.Equal(2, curriculum.StageIndex);
            Assert.True(curriculum.IsAtFinalStage);
        }

        [Fact]
        public void Constructor_RestoresStageAndGenerationsFromCheckpoint()
        {
            var curriculum = new MarioCurriculum(
                stageCount: 4,
                windowSize: 5,
                advanceThresholdPercent: 50f,
                startingStageIndex: 2,
                startingGenerationsAtStage: 7);

            Assert.Equal(2, curriculum.StageIndex);
            Assert.Equal(7, curriculum.GenerationsAtStage);
            Assert.False(curriculum.IsAtFinalStage);
        }

        [Fact]
        public void Constructor_ClampsOutOfRangeStartingStageIndex()
        {
            var curriculum = new MarioCurriculum(stageCount: 3, startingStageIndex: 99);

            Assert.Equal(2, curriculum.StageIndex);
            Assert.True(curriculum.IsAtFinalStage);
        }
    }
}
