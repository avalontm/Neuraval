using System;
using Neuraval.Core.Models.RoPE;
using Neuraval.Tensor;

namespace Neuraval.Core.Models
{
    public class ModernDecoderBlock
    {
        private readonly TransformerConfig _config;
        private readonly int _seed;

        private RMSNorm _norm1;
        private RMSNorm _norm2;
        private RotaryEmbedding _rotary;
        private GQAAttention _attention;
        private SwiGLUFeedForward _feedforward;

        private float[,,]? _lastInput;
        private float[,,]? _lastResidual1;

        public int HiddenSize => _config.HiddenSize;
        public int NumAttentionHeads => _config.NumAttentionHeads;
        public int NumKeyValueHeads => _config.NumKeyValueHeads;
        public int IntermediateSize => _config.IntermediateSize;

        public ModernDecoderBlock(TransformerConfig config, int seed = 42)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            config.Validate();

            _config = config.Clone();
            _seed = seed;

            _norm1 = new RMSNorm(_config.HiddenSize, _config.RmsNormEps);
            _norm2 = new RMSNorm(_config.HiddenSize, _config.RmsNormEps);

            _rotary = new RotaryEmbedding(new RotaryConfig
            {
                HeadDim = _config.HeadDim,
                MaxPositionEmbeddings = _config.MaxPositionEmbeddings,
                RopeTheta = _config.RopeTheta
            });

            _attention = new GQAAttention(_config.HiddenSize, _config.NumAttentionHeads, _config.NumKeyValueHeads, _rotary, seed);
            _feedforward = new SwiGLUFeedForward(_config.HiddenSize, _config.IntermediateSize, seed + 1);
        }

        public void ZeroGradients()
        {
            _norm1.ZeroGradients();
            _norm2.ZeroGradients();
            _attention.ZeroGradients();
            _feedforward.ZeroGradients();
        }

        public void AverageGradients(int batchSize)
        {
            _norm1.AverageGradients(batchSize);
            _norm2.AverageGradients(batchSize);
            _attention.AverageGradients(batchSize);
            _feedforward.AverageGradients(batchSize);
        }

        public float[,,] Forward(float[,,] input, int positionOffset = 0)
        {
            int embDim = input.GetLength(2);

            if (embDim != _config.HiddenSize)
            {
                throw new ArgumentException($"Input dimension {embDim} does not match expected {_config.HiddenSize}");
            }

            _lastInput = (float[,,])input.Clone();

            var norm1Output = _norm1.ForwardBatch(input);
            var attentionOutput = _attention.Forward(norm1Output, positionOffset);
            var residual1 = AddResidualBatch(input, attentionOutput);

            _lastResidual1 = residual1;

            var norm2Output = _norm2.ForwardBatch(residual1);
            var ffOutput = _feedforward.ForwardBatch(norm2Output);
            var output = AddResidualBatch(residual1, ffOutput);

            return output;
        }

        public float[,,] ForwardIncremental(float[,,] input, int positionOffset, GqaKeyValueCacheLayer cache)
        {
            int embDim = input.GetLength(2);

            if (embDim != _config.HiddenSize)
                throw new ArgumentException($"Input dimension {embDim} does not match expected {_config.HiddenSize}");

            var norm1Output = _norm1.ForwardBatch(input);
            var attentionOutput = _attention.ForwardIncremental(norm1Output, positionOffset, cache);
            var residual1 = AddResidualBatch(input, attentionOutput);

            var norm2Output = _norm2.ForwardBatch(residual1);
            var ffOutput = _feedforward.ForwardBatch(norm2Output);
            var output = AddResidualBatch(residual1, ffOutput);

            return output;
        }

        public float[,,] Backward(float[,,] gradOutput)
        {
            if (_lastInput == null || _lastResidual1 == null)
                throw new InvalidOperationException("Forward must be called before Backward");

            var gradNorm2Output = _feedforward.BackwardBatch(gradOutput, 0f);
            var gradResidual1FromNorm2 = _norm2.BackwardBatch(gradNorm2Output);
            var gradResidual1 = AddResidualBatch(gradOutput, gradResidual1FromNorm2);

            var gradNorm1Output = _attention.Backward(gradResidual1);
            var gradInputFromNorm1 = _norm1.BackwardBatch(gradNorm1Output);
            var gradInput = AddResidualBatch(gradResidual1, gradInputFromNorm1);

            return gradInput;
        }

        private static float[,,] AddResidualBatch(float[,,] input, float[,,] residual)
        {
            var inputTensor = Neuraval.Tensor.Tensor.FromArray3D(input);
            var residualTensor = Neuraval.Tensor.Tensor.FromArray3D(residual);

            return TensorOps.Add(inputTensor, residualTensor).ToArray3D();
        }

        public void UpdateWeights(float learningRate)
        {
            _norm1.UpdateWeights(learningRate);
            _norm2.UpdateWeights(learningRate);
            _attention.UpdateWeights(learningRate);
            _feedforward.UpdateWeights(learningRate);
        }

        public void ResetGradients()
        {
            _norm1.ResetGradients();
            _norm2.ResetGradients();
            _attention.ZeroGradients();
            _feedforward.ResetGradients();
        }

        public ModernDecoderBlockState SaveState()
        {
            return new ModernDecoderBlockState
            {
                Config = _config.Clone(),
                Seed = _seed,
                Norm1State = _norm1.SaveState(),
                Norm2State = _norm2.SaveState(),
                AttentionState = _attention.SaveState(),
                FeedforwardState = _feedforward.SaveState()
            };
        }

        public static ModernDecoderBlock LoadState(ModernDecoderBlockState state)
        {
            var block = new ModernDecoderBlock(state.Config, state.Seed);

            block._norm1 = RMSNorm.LoadState(state.Norm1State);
            block._norm2 = RMSNorm.LoadState(state.Norm2State);
            block._attention = GQAAttention.LoadState(state.AttentionState, block._rotary);
            block._feedforward = SwiGLUFeedForward.LoadState(state.FeedforwardState);

            return block;
        }
    }

    public class ModernDecoderBlockState
    {
        public TransformerConfig Config { get; set; } = null!;
        public int Seed { get; set; }
        public RMSNormState Norm1State { get; set; } = null!;
        public RMSNormState Norm2State { get; set; } = null!;
        public GQAAttentionState AttentionState { get; set; } = null!;
        public SwiGLUFeedForwardState FeedforwardState { get; set; } = null!;
    }
}
