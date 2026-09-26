using System.Text.Json.Serialization;

namespace Neuraval.Core.Serialization.ModelExport
{
    public sealed class GenerationConfigDocument
    {
        [JsonPropertyName("temperature")]
        public float Temperature { get; set; }

        [JsonPropertyName("top_p")]
        public float TopP { get; set; }

        [JsonPropertyName("top_k")]
        public int TopK { get; set; }

        [JsonPropertyName("max_new_tokens")]
        public int MaxNewTokens { get; set; }
    }
}
