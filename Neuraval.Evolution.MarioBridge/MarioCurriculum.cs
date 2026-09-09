using System;
using System.Collections.Generic;
using System.Linq;

namespace Neuraval.Evolution.MarioBridge
{
    public sealed class MarioCurriculum
    {
        public const int DefaultWindowSize = 8;
        public const float DefaultAdvanceThresholdPercent = 50f;

        private readonly int _stageCount;
        private readonly int _windowSize;
        private readonly float _advanceThresholdPercent;
        private readonly Queue<float> _window = new();

        public int StageIndex { get; private set; }
        public int GenerationsAtStage { get; private set; }

        public bool IsAtFinalStage => StageIndex >= _stageCount - 1;

        public int WindowSampleCount => _window.Count;
        public int WindowCapacity => _windowSize;
        public float WindowAverageCompletion => _window.Count == 0 ? 0f : _window.Average();

        public MarioCurriculum(
            int stageCount,
            int windowSize = DefaultWindowSize,
            float advanceThresholdPercent = DefaultAdvanceThresholdPercent,
            int startingStageIndex = 0,
            int startingGenerationsAtStage = 0)
        {
            if (stageCount < 1)
            {
                throw new ArgumentOutOfRangeException(nameof(stageCount), "Debe haber al menos un tramo configurado.");
            }

            _stageCount = stageCount;
            _windowSize = Math.Max(1, windowSize);
            _advanceThresholdPercent = advanceThresholdPercent;
            StageIndex = Math.Clamp(startingStageIndex, 0, stageCount - 1);
            GenerationsAtStage = Math.Max(0, startingGenerationsAtStage);
        }

        public bool RecordGeneration(float completionPercent)
        {
            GenerationsAtStage++;

            if (IsAtFinalStage)
            {
                _window.Enqueue(completionPercent);
                TrimWindow();
                return false;
            }

            _window.Enqueue(completionPercent);
            TrimWindow();

            if (_window.Count < _windowSize)
            {
                return false;
            }

            if (WindowAverageCompletion < _advanceThresholdPercent)
            {
                return false;
            }

            StageIndex++;
            GenerationsAtStage = 0;
            _window.Clear();
            return true;
        }

        private void TrimWindow()
        {
            while (_window.Count > _windowSize)
            {
                _window.Dequeue();
            }
        }
    }
}
