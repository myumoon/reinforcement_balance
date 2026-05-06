from base.entity_attention_extractor import EntityAttentionExtractor


class CoinEntityAttentionExtractor(EntityAttentionExtractor):
    def __init__(self, observation_space, features_dim=128, offsets=None):
        super().__init__(
            observation_space,
            features_dim=features_dim,
            offsets=offsets,
            item_key="coins",
            use_polar=True,
            dist_alpha=0.0,
        )
