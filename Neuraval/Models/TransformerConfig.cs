using System;

namespace Neuraval.Core.Models
{
    public sealed class TransformerConfig
    {
        public string Architecture { get; set; } = "legacy";

        public int VocabSize { get; set; }
        public int HiddenSize { get; set; }

        public int NumHiddenLayers { get; set; }

        public int NumAttentionHeads { get; set; }
        public int NumKeyValueHeads { get; set; }

        public int IntermediateSize { get; set; }

        public int MaxPositionEmbeddings { get; set; }

        public float RopeTheta { get; set; } = 10000f;
        public float RmsNormEps { get; set; } = 1e-6f;

        public string Activation { get; set; } = "silu";
        public string NormType { get; set; } = "rmsnorm";

        public bool AttentionBias { get; set; }
        public bool TieWordEmbeddings { get; set; }

        public float Dropout { get; set; } = 0.1f;

        public int Seed { get; set; } = 42;

        public void Validate()
        {
            if (VocabSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(VocabSize));

            if (HiddenSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(HiddenSize));

            if (NumHiddenLayers <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumHiddenLayers));

            if (NumAttentionHeads <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumAttentionHeads));

            if (NumKeyValueHeads <= 0)
                throw new ArgumentOutOfRangeException(nameof(NumKeyValueHeads));

            if (IntermediateSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(IntermediateSize));

            if (MaxPositionEmbeddings <= 0)
                throw new ArgumentOutOfRangeException(nameof(MaxPositionEmbeddings));

            if (HiddenSize % NumAttentionHeads != 0)
                throw new InvalidOperationException(
                    $"HiddenSize ({HiddenSize}) debe ser divisible por NumAttentionHeads ({NumAttentionHeads})");

            if (NumAttentionHeads % NumKeyValueHeads != 0)
                throw new InvalidOperationException(
                    $"NumAttentionHeads ({NumAttentionHeads}) debe ser divisible por NumKeyValueHeads ({NumKeyValueHeads})");

            if (RmsNormEps <= 0f)
                throw new InvalidOperationException("RmsNormEps debe ser positivo");

            if (RopeTheta <= 0f)
                throw new InvalidOperationException("RopeTheta debe ser positivo");

            if (Dropout < 0f || Dropout >= 1f)
                throw new InvalidOperationException("Dropout debe estar en el rango [0, 1)");
        }

        public int HeadDim => HiddenSize / NumAttentionHeads;

        public int NumKeyValueGroups => NumAttentionHeads / NumKeyValueHeads;

        public TransformerConfig Clone()
        {
            return new TransformerConfig
            {
                Architecture = Architecture,
                VocabSize = VocabSize,
                HiddenSize = HiddenSize,
                NumHiddenLayers = NumHiddenLayers,
                NumAttentionHeads = NumAttentionHeads,
                NumKeyValueHeads = NumKeyValueHeads,
                IntermediateSize = IntermediateSize,
                MaxPositionEmbeddings = MaxPositionEmbeddings,
                RopeTheta = RopeTheta,
                RmsNormEps = RmsNormEps,
                Activation = Activation,
                NormType = NormType,
                AttentionBias = AttentionBias,
                TieWordEmbeddings = TieWordEmbeddings,
                Dropout = Dropout,
                Seed = Seed
            };
        }
    }
}
