using System.Text.Json.Serialization;

namespace Neuraval.Core.Serialization.ModelExport
{
    public sealed class ModelConfigDocument
    {
        [JsonPropertyName("architecture")]
        public string Architecture { get; set; } = "legacy";

        [JsonPropertyName("vocab_size")]
        public int VocabSize { get; set; }

        [JsonPropertyName("hidden_size")]
        public int HiddenSize { get; set; }

        [JsonPropertyName("num_hidden_layers")]
        public int NumHiddenLayers { get; set; }

        [JsonPropertyName("num_attention_heads")]
        public int NumAttentionHeads { get; set; }

        [JsonPropertyName("num_key_value_heads")]
        public int NumKeyValueHeads { get; set; }

        [JsonPropertyName("intermediate_size")]
        public int IntermediateSize { get; set; }

        [JsonPropertyName("max_position_embeddings")]
        public int MaxPositionEmbeddings { get; set; }

        [JsonPropertyName("rope_theta")]
        public float RopeTheta { get; set; }

        [JsonPropertyName("rms_norm_eps")]
        public float RmsNormEps { get; set; }

        [JsonPropertyName("hidden_act")]
        public string Activation { get; set; } = "silu";

        [JsonPropertyName("norm_type")]
        public string NormType { get; set; } = "rmsnorm";

        [JsonPropertyName("attention_bias")]
        public bool AttentionBias { get; set; }

        [JsonPropertyName("tie_word_embeddings")]
        public bool TieWordEmbeddings { get; set; }
    }
}
