namespace Neuraval.Core.Serialization.WeightLoading
{
    public enum TensorRole
    {
        EmbedTokens,
        InputLayerNorm,
        SelfAttnQProj,
        SelfAttnKProj,
        SelfAttnVProj,
        SelfAttnOProj,
        PostAttentionLayerNorm,
        MlpGateProj,
        MlpUpProj,
        MlpDownProj,
        FinalNorm,
        LmHead
    }
}
