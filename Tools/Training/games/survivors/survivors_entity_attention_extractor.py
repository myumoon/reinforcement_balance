from base.entity_attention_extractor import EntityAttentionExtractor


class SurvivorsEntityAttentionExtractor(EntityAttentionExtractor):
    def __init__(self, observation_space, features_dim=128, offsets=None):
        super().__init__(
            observation_space,
            features_dim=features_dim,
            offsets=offsets,
            item_key="gem_rel_pos",
            use_polar=True,
            enemy_scalar_keys=["enemy_type", "enemy_hp"],
        )
