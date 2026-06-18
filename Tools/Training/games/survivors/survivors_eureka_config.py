"""Survivors ゲーム専用の EUREKA 設定。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from base.eureka_game_config import EurekaGameConfig
from common.obs_schema import fetch_obs_schema, build_obs_layout
from games.survivors.survivors_reward_validator import validate_survivors_reward_code
from games.survivors.survivors_source_of_truth import (
    build_survivors_source_of_truth,
    render_source_of_truth_markdown,
)
from games.survivors.survivors_weapon_curriculum import WeaponType


class SurvivorsEurekaConfig(EurekaGameConfig):
    """Survivors ゲーム専用の EUREKA 設定。"""

    def __init__(self):
        self._obs_layout_str: str = ""
        self._offsets: dict[str, int] = {}
        self._obs_schema: dict | None = None
        self._source_of_truth: dict | None = None

    def setup(self, host: str, port: int) -> None:
        print(f"[INFO] obs_schema を取得中 ({host}:{port})...")
        schema = fetch_obs_schema(host, port)
        self._obs_schema = schema
        self._obs_layout_str, self._offsets = build_obs_layout(schema)
        print(f"[INFO] obs_schema 取得完了: total_dim={schema['total_dim']}")

    def make_env(self, host: str, port: int, frame_skip: int = 1):
        from games.survivors.survivors_env import SurvivorsEnv
        return SurvivorsEnv(host=host, port=port, frame_skip=frame_skip)

    def build_source_of_truth(self, host: str, port: int) -> dict:
        repo_root = Path(__file__).resolve().parents[4]
        self._source_of_truth = build_survivors_source_of_truth(
            repo_root=repo_root,
            host=host,
            port=port,
            obs_schema=self._obs_schema,
        )
        return self._source_of_truth

    def render_source_of_truth(self, source_of_truth: dict | None) -> str:
        return render_source_of_truth_markdown(source_of_truth)

    def validate_reward_code(self, code: str, source_of_truth: dict | None) -> list[dict]:
        return validate_survivors_reward_code(code, source_of_truth)

    def _source_of_truth_section(self, source_of_truth: dict | None) -> str:
        rendered = self.render_source_of_truth(source_of_truth)
        if not rendered:
            return ""
        return self._titled_section("Source of Truth (C++/UE5 auto extracted)", rendered)

    def _effective_source_of_truth(self, source_of_truth: dict | None = None) -> dict:
        return source_of_truth or self._source_of_truth or {}

    def _reward_constant(self, name: str, default: float) -> float:
        return float(
            self._effective_source_of_truth().get("reward_constants", {}).get(name, default)
        )

    def _player_constant(self, name: str, default: float) -> float:
        return float(
            self._effective_source_of_truth().get("player_constants", {}).get(name, default)
        )

    def _observation_constant(self, name: str, default: float | int) -> float | int:
        return self._effective_source_of_truth().get("observation_constants", {}).get(name, default)

    def _garlic_summary(self) -> str:
        table = self._effective_source_of_truth().get("garlic_table", [])
        if not table:
            return "GarlicTable は Source of Truth のC++ snippetを参照してください。"
        first = table[0]
        last = table[-1]
        return (
            f"GarlicTable: Lv1 damage={first['damage']}, interval={first['hit_interval']}s, "
            f"radius={first['area_radius']}u / Lv{len(table)} damage={last['damage']}, "
            f"interval={last['hit_interval']}s, radius={last['area_radius']}u"
        )

    def _gem_xp_summary(self) -> str:
        values = self._effective_source_of_truth().get("gem_xp_values", [])
        return f"GemXPValues: {values}" if values else "GemXPValues は Source of Truth のC++ snippetを参照してください。"

    def _enemy_summary(self) -> str:
        enemies = self._effective_source_of_truth().get("enemy_types", [])
        if not enemies:
            return "EnemyTypeTable は Source of Truth のC++ snippetを参照してください。"
        rows = [
            f"TypeId {e['type_id']} {e['name']}: HP={e['base_hp']}, speed={e['speed']}, damage={e['contact_damage']}"
            for e in enemies
        ]
        return "\n".join(rows)

    # ------------------------------------------------------------------ #
    # プロンプトセクション                                                  #
    # ------------------------------------------------------------------ #

    def _prompt_section_game_overview(self) -> str:
        item_reward = self._reward_constant("ItemReward", 1.0)
        kill_reward = self._reward_constant("KillReward", 2.0)
        return (
            f"2D フィールド（±15m の正方形）でプレイヤーが敵を倒しながらアイテムを集めて生き延びるゲームです。\n"
            f"- プレイヤーは離散9方向（0=北, 1=北東, 2=東, 3=南東, 4=南, 5=南西, 6=西, 7=北西, 8=静止）で移動\n"
            f"- プレイヤーは weapon_slots の武器（近距離オーラ・方向性発射・エリア攻撃・防御シールド等）で自動攻撃し、敵を倒せる\n"
            f"- 敵に接触すると毎ティック HP が減少し、HP=0 でエピソード終了\n"
            f"- 敵はフィールド外周からスポーンし、プレイヤーに向かって移動する（同時敵数とスポーン頻度はカリキュラムで変化）\n"
            f"- 敵を倒すとGemがドロップし、Gem取得でXPとItemRewardを得る\n"
            f"- Gem取得（ItemReward={item_reward}）と敵撃破（KillReward={kill_reward}）で得点を稼ぐ\n"
            f"- XPRequiredForLevel、GemXPValues、GarlicTable は Source of Truth のC++抽出値を参照すること\n"
            f"- レベルアップでGarlicが強化される。手書きのXP式やGarlic式を仮定しないこと\n"
        )

    def _prompt_section_game_objective(self) -> str:
        item_reward = self._reward_constant("ItemReward", 1.0)
        kill_reward = self._reward_constant("KillReward", 2.0)
        return (
            f"**最優先目標**: 1エピソードで「アイテムを取り敵を倒しながら生き延びる」こと。\n"
            f"- 死ぬとエピソード終了するため、HP 管理が最重要\n"
            f"- ただし壁際に逃げ続けるだけでは得点が低い\n"
            f"- 装備武器の特性（近距離/遠距離/エリア）に応じた間合いで敵を処理しつつアイテムを取る積極的行動が最善\n"
            f"- **評価指標: item_kill_score** = (base_reward - AliveReward×ep_len) の平均\n"
            f"  純粋なGem+Kill スコア。生存だけでは 0.0、Gem1個で {item_reward:.1f}、Kill1体で {kill_reward:.1f}"
        )

    def _prompt_section_obs_index(self) -> str:
        # 740次元 obs スキーマ（PR2 全武器対応版）に合わせた説明を生成する。
        # セグメント開始オフセットは _offsets dict（実 C++ schema から取得）を優先し、
        # 取得できない場合は PR2 schema の既定値にフォールバックする。
        o = self._offsets
        max_player_hp = self._player_constant("MaxPlayerHP", 100.0)
        item_reward = self._reward_constant("ItemReward", 1.0)
        max_player_level = self._observation_constant("MaxPlayerLevel", 100)

        # --- PR2 schema オフセット（デフォルト値は schema 計算値） ---
        hp_i        = o.get("player_hp",               12)
        wpn_i       = o.get("weapon_slots",             23)   # 18 dims: (type,level,cd)×6
        psvslot_i   = o.get("passive_slots",            41)   # 12 dims: (type,level)×6
        ecnt_i      = o.get("enemy_count",              53)
        elapsed_i   = o.get("elapsed_time",             54)
        xp_i        = o.get("xp_progress",              55)
        plvl_i      = o.get("player_level",             56)
        red_gem_i   = o.get("red_gem_rel_pos",          58)   # 20 dims: ×10 gems
        grn_gem_i   = o.get("green_gem_rel_pos",        78)   # 24 dims: ×12 gems
        blu_gem_i   = o.get("blue_gem_rel_pos",        102)   # 24 dims: ×12 gems
        gem_pr_i    = o.get("gem_pickup_radius",       126)   # 1 dim
        er_i        = o.get("enemy_rel_pos",           127)   # 64 dims: ×32 enemies
        ev_i        = o.get("enemy_vel",               191)   # 64 dims
        et_i        = o.get("enemy_type",              255)   # 32 dims
        ehp_i       = o.get("enemy_hp",               287)   # 32 dims
        efz_i       = o.get("enemy_frozen",            319)   # 32 dims
        end_i       = o.get("enemy_nearest_dist_16dir", 351)
        end_dn_i    = o.get("enemy_density_near_16dir", 367)
        end_dm_i    = o.get("enemy_density_mid_16dir",  383)
        gem_da_i    = o.get("gem_density_all_16dir",   399)   # 48 dims
        gem_drg_i   = o.get("red_green_gem_density_16dir", 447)  # 48 dims
        proj_i      = o.get("projectiles",             495)   # 192 dims: (dx,dy,r,vx,vy,warning)×32
        fp_i        = o.get("floor_pickups",           687)   # 24 dims
        sp_i        = o.get("special_pickups",         711)   # 9 dims
        dest_i      = o.get("destructibles",           720)   # 20 dims

        max_enemy   = (ev_i - er_i) // 2   # = 32

        return (
            f"**obs 合計: 740 次元（PR2 全武器・パッシブ対応スキーマ）**\n"
            f"\n"
            f"**プレイヤー状態**\n"
            f"  obs[0:2]    = player_pos (x,y) / FieldHalfSize → [-1, 1]\n"
            f"  obs[2:4]    = player_vel (vx,vy) / MoveSpeed\n"
            f"  obs[4:12]   = wall_rays 8方向 (0=壁近い, 1=遠い)\n"
            f"  obs[{hp_i}]      = player_hp / MaxPlayerHP={max_player_hp} (0~1)\n"
            f"  obs[13]     = shield_active (0/1, Laurel シールド中=1)\n"
            f"  obs[14]     = shield_timer_norm (残シールド時間 0~1)\n"
            f"  obs[15]     = revival_remaining_norm (Tirajisu リバイバル残数 0~1)\n"
            f"  obs[16]     = armor_flat_norm (Armor パッシブ合計 0~1)\n"
            f"  obs[17]     = regen_per_sec_norm (Pummarola 回復量 0~1)\n"
            f"  obs[18:23]  = passive_effect_summary 5次元: "
            f"(damage_mult, cooldown_reduction, area_mult, move_speed_mult, magnet_mult) 各0~1\n"
            f"\n"
            f"**武器・パッシブスロット**\n"
            f"  obs[{wpn_i}:{wpn_i+18}] = weapon_slots × 6スロット: (type_norm, level_norm, cooldown_norm) 各 [0,1]\n"
            f"    type_norm  = EWeaponType_id / 64 (Garlic=1/64≈0.016, None=0)\n"
            f"    level_norm = level / 8\n"
            f"    cooldown_norm = 残クールダウン / max_cooldown\n"
            f"    空スロット = (0, 0, 0)\n"
            f"  obs[{psvslot_i}:{psvslot_i+12}] = passive_slots × 6スロット: (type_norm, level_norm) 各 [0,1]\n"
            f"    type_norm = EPassiveItemType_id / 32\n"
            f"\n"
            f"**ゲーム進行**\n"
            f"  obs[{ecnt_i}]     = 敵数 / {max_enemy}\n"
            f"  obs[{elapsed_i}]     = elapsed_time / MaxGameTime (0~1)\n"
            f"  obs[{xp_i}]     = xp_progress (0~1): レベルアップ時 0 にリセット\n"
            f"  obs[{plvl_i}]     = player_level / {max_player_level} (0~1)\n"
            f"  obs[57]     = stage_id_norm (0=MadForest)\n"
            f"\n"
            f"**Gem 相対位置（距離昇順、未出現スロットは 0 パディング）**\n"
            f"  obs[{red_gem_i}:{red_gem_i+20}]  = red_gem_rel_pos: 最近傍 Red ×10 (dx,dy)/DN\n"
            f"  obs[{grn_gem_i}:{grn_gem_i+24}]  = green_gem_rel_pos: 最近傍 Green ×12 (dx,dy)/DN\n"
            f"  obs[{blu_gem_i}:{blu_gem_i+24}]  = blue_gem_rel_pos: 最近傍 Blue ×12 (dx,dy)/DN\n"
            f"  obs[{gem_pr_i}]   = gem_pickup_radius / MaxGemPickupRadius (Attractorb で増加)\n"
            f"\n"
            f"**敵情報（i=0 が最近傍、最大{max_enemy}体、未出現スロットは 0 パディング）**\n"
            f"  obs[{er_i}+i*2 : {er_i}+i*2+2] = enemy_rel_pos: (dx,dy)/DN\n"
            f"  obs[{ev_i}+i*2 : {ev_i}+i*2+2] = enemy_vel: (vx,vy)/MoveSpeed\n"
            f"  obs[{et_i}+i]    = enemy_type (0~1)\n"
            f"  obs[{ehp_i}+i]   = enemy_hp (0~1, 0=瀕死)\n"
            f"  obs[{efz_i}+i]   = enemy_frozen (0/1, Orologion フリーズ中=1)\n"
            f"\n"
            f"**方向別密度/最近傍距離（16方向）**\n"
            f"  obs[{end_i}:{end_i+16}]   = enemy_nearest_dist_16dir (0=危険, 1=安全)\n"
            f"  obs[{end_dn_i}:{end_dn_i+16}]  = enemy_density_near_16dir (0~6m密度)\n"
            f"  obs[{end_dm_i}:{end_dm_i+16}]  = enemy_density_mid_16dir (6~14m密度)\n"
            f"  obs[{gem_da_i}:{gem_da_i+48}]   = gem_density_all_16dir (全Gem ×3密度特徴量)\n"
            f"  obs[{gem_drg_i}:{gem_drg_i+48}]  = red_green_gem_density_16dir (Red+Green ×3)\n"
            f"\n"
            f"**プロジェクタイル（武器弾・GroundZone 混在、Level高い順→距離近い順）**\n"
            f"  obs[{proj_i}:{proj_i+192}] = projectiles ×32: (dx,dy,radius_norm,vx_norm,vy_norm,warning) 各6次元\n"
            f"    warning=1 は Santa Water / La Borra の着弾予兆、未使用スロット = (0,0,0,0,0,0)\n"
            f"\n"
            f"**フロアアイテム・特殊アイテム・破壊物**\n"
            f"  obs[{fp_i}:{fp_i+24}]   = floor_pickups ×8: (dx,dy,type_norm) 各3次元\n"
            f"  obs[{sp_i}:{sp_i+9}]    = special_pickups ×3: (dx,dy,type_norm)\n"
            f"  obs[{dest_i}:{dest_i+20}]  = destructibles ×10: (dx,dy)\n"
            f"\n"
            f"**Gem 取得検知**\n"
            f"  obs[{xp_i}] > prev_obs[{xp_i}] または base_reward >= {item_reward}\n"
            f"  （レベルアップ直後は obs[{xp_i}] が 0 に戻るため base_reward 判定を推奨）\n"
            f"\n"
            f"**⚠ 注意事項**\n"
            f"  weapon_slots の type_norm は武器フェーズにより変化する（W0: Garlic のみ、W6: 全武器）\n"
            f"  reward_fn は特定武器の固定値を仮定しないこと（obs スキーマは永続固定だが武器構成は可変）\n"
            f"  Garlic等の武器パラメータは Source of Truth のC++ snippetを参照すること"
        )

    def _prompt_section_fixed_rewards(self) -> str:
        alive_reward = self._reward_constant("AliveReward", 0.001)
        item_reward = self._reward_constant("ItemReward", 1.0)
        kill_reward = self._reward_constant("KillReward", 2.0)
        return (
            f"- AliveReward = {alive_reward} / step（生存毎ステップ）\n"
            f"- ItemReward  = {item_reward}（Gem取得時）\n"
            f"- KillReward  = {kill_reward}（敵撃破時）\n"
            f"- {self._gem_xp_summary()}\n"
            f"- XPRequiredForLevel は Source of Truth のC++ snippetを参照してください。"
        )

    def _prompt_section_physics(self) -> str:
        max_player_hp = self._player_constant("MaxPlayerHP", 100.0)
        move_speed = self._player_constant("MoveSpeed", 80.0)
        gem_pickup_radius = self._player_constant("GemPickupRadius", 50.0)
        contact_interval = self._observation_constant("ContactHitInterval", "Source of Truth")
        return (
            f"- プレイヤー最大 HP: {max_player_hp}\n"
            f"- プレイヤー移動速度: {move_speed}\n"
            f"- Gem pickup radius: {gem_pickup_radius}\n"
            f"- 接触ダメージ間隔: {contact_interval}\n"
            f"- {self._garlic_summary()}\n"
            f"- 敵タイプ別パラメータ:\n{self._enemy_summary()}\n"
            f"- スポーン間隔と同時敵数はカリキュラム/paramsで変化するため、固定のMaxActiveEnemiesを前提にしないこと"
        )

    def _prompt_section_weapon_profiles(self) -> str:
        wpn_i = self._offsets.get("weapon_slots", 23)
        W = WeaponType
        melee   = [W.GARLIC, W.WHIP, W.KING_BIBLE, W.SOUL_EATER, W.BLOODY_TEAR, W.UNHOLY_VESPERS]
        ranged  = [W.MAGIC_WAND, W.KNIFE, W.AXE, W.CROSS, W.FIRE_WAND,
                   W.PEACHONE, W.EBONY_WINGS, W.HOLY_WAND, W.THOUSAND_EDGE,
                   W.DEATH_SPIRAL, W.HEAVEN_SWORD, W.HELLFIRE, W.NO_FUTURE, W.VANDALIER]
        area    = [W.SANTA_WATER, W.RUNETRACER, W.LIGHTNING_RING, W.PENTAGRAM,
                   W.LA_BORRA, W.THUNDER_LOOP, W.GORGEOUS_MOON]
        defensive = [W.LAUREL]

        def ids_str(ids):
            return ", ".join(str(i) for i in ids)

        def norms_str(ids):
            return ", ".join(f"{i/64:.3f}" for i in ids)

        return (
            f"reward_fn は obs[{wpn_i}:{wpn_i+18}] (weapon_slots, 6スロット×3要素) を読み取り、\n"
            f"装備中の武器タイプから立ち回り傾向を推定して、敵密度・距離シェーピングの重みを変えること。\n"
            f"各スロットの type_norm = EWeaponType_id / 64。type_norm=0 は空スロット。\n"
            f"\n"
            f"武器カテゴリと EWeaponType_id / type_norm:\n"
            f"\n"
            f"**近距離型 (melee/aura)** — ID: {ids_str(melee)}\n"
            f"  type_norm: {norms_str(melee)}\n"
            f"  立ち回り: 敵を安全な密度範囲（過密でない）で引き付けてGemを回収。\n"
            f"           近距離敵接近への過剰なペナルティは避け、適度な巻き込みを許容すること。\n"
            f"\n"
            f"**遠距離・方向性型 (ranged/directional)** — ID: {ids_str(ranged)}\n"
            f"  type_norm: {norms_str(ranged)}\n"
            f"  立ち回り: 敵との中距離維持を評価。近距離敵密度が高い方向を近距離型より強く避けること。\n"
            f"           カイト行動（敵群の外縁から攻撃しながら移動）を小さく補助してよい。\n"
            f"\n"
            f"**エリア・ゾーン型 (area/zone)** — ID: {ids_str(area)}\n"
            f"  type_norm: {norms_str(area)}\n"
            f"  立ち回り: 敵群の外周を回りながらGemを回収。ゾーン攻撃が届く範囲に敵を誘導しつつ、\n"
            f"           過密方向への突入は避けること。中距離維持が基本。\n"
            f"\n"
            f"**防御型 (defensive)** — ID: {ids_str(defensive)}\n"
            f"  type_norm: {norms_str(defensive)}\n"
            f"  立ち回り: shield_active (obs[13]=1) 時はGem回収を強め、\n"
            f"           shield_timer_norm (obs[14]) が低い時は安全回避を優先すること。\n"
            f"\n"
            f"実装指針:\n"
            f"- 各スロットの type_norm (0は空スロット) を上記カテゴリ範囲テーブルに照合し、\n"
            f"  カテゴリ別の占有スロット数を集計すること。最多占有カテゴリを主力として採用する。\n"
            f"  例: melee_count=3, ranged_count=1, area_count=2 → 主力=melee\n"
            f"- カテゴリが混在する場合は、各カテゴリの占有スロット数を総スロット数で割った比率で\n"
            f"  敵密度ペナルティ重みを按分すること。\n"
            f"  例: melee=3/6=0.5, ranged=2/6=0.33, area=1/6=0.17 の場合、\n"
            f"  penalty_weight = melee_w * 0.5 + ranged_w * 0.33 + area_w * 0.17\n"
            f"- 全武器に同一の敵距離・密度ペナルティ係数を適用しないこと。\n"
            f"- 武器カテゴリごとの大きな固定報酬オフセットは加えず、\n"
            f"  Gem接近・敵密度回避・間合い維持の重み係数だけを小さく変えること。\n"
            f"- 近距離型: 安全密度閾値を低め（近い敵をより許容）に設定。\n"
            f"- 遠距離型: 安全密度閾値を高め（敵密度に敏感）に設定。\n"
            f"- エリア型: 中間的な設定でゾーン制圧行動を評価。\n"
        )

    def _prompt_section_scale_constraints(self) -> str:
        max_hp_penalty = self._player_constant("MaxPlayerHP", 100.0)
        return (
            f"- **HP ペナルティは survivors_env が永続的に適用済み**（info['hp_penalty']）\n"
            f"  reward_fn でさらに HP 差分ペナルティを追加しないこと（二重計上になる）\n"
            f"  HP 状態を使う場合は「obs[12] が低い時にアイテム接近を促す」など間接的な利用にとどめること\n"
            f"\n"
            f"- **HP ペナルティの累積スケール（重要）**\n"
            f"  1HP ダメージ/step → hp_penalty = -1.0、最大 -{max_hp_penalty:.0f} / エピソード（全 HP 消費時）\n"
            f"  ep_rew_mean(SB3) が大きく負になる主因は hp_penalty であり、shaped_reward ではない\n"
            f"  hp_penalty_mean を見て shaped_reward_mean の貢献を判断すること\n"
            f"  例: hp_penalty_mean=-80, shaped_reward_mean=-5 なら shaped はほぼ影響なし\n"
            f"\n"
            f"- 敵接近ペナルティは [-0.05, 0.0] 程度まで\n"
            f"- アイテム接近ボーナスは 1ステップあたり [-0.03, 0.03] 程度まで（アイテム10個に対して設計）\n"
            f"  Gem 距離計算は obs schema の segment offset（red_gem_rel_pos 等）を Source of Truth から取得して使うこと\n"
            f"- Gem 回収は待機による偶然取得ではなく、最近 Gem への距離短縮と取得後の次 Gem 追従を明示的に評価すること\n"
            f"- 敵が多い場合は、Gem 方向へ直進するだけでなく、敵密度が低い方向から回り込んで Gem に近づく行動を評価すること\n"
            f"- item_kill_score = 0 は「生存のみ」。reward_fn は item_kill_score を上げることを目標とすること\n"
            f"- エピソード全体の shaped_reward 累計が base_reward を大幅に超えないよう設計すること"
        )

    # ------------------------------------------------------------------ #
    # ゲームコンテキスト・プロンプト構築                                      #
    # ------------------------------------------------------------------ #

    def build_game_context(self, source_of_truth: dict | None = None) -> str:
        return "\n\n".join([
            self._source_of_truth_section(source_of_truth),
            self._titled_section("ゲーム概要",           self._prompt_section_game_overview()),
            self._titled_section("ゲームの目標",          self._prompt_section_game_objective()),
            self._titled_section("obs インデックス一覧",  self._prompt_section_obs_index()),
            self._titled_section("固定報酬（C++ 側）",    self._prompt_section_fixed_rewards()),
            self._titled_section("物理定数",              self._prompt_section_physics()),
            self._titled_section("武器タイプ別立ち回り指針", self._prompt_section_weapon_profiles()),
            self._titled_section("スケール制約",           self._prompt_section_scale_constraints()),
            self._titled_section("メトリクスの意味",       self.metrics_description()),
        ])

    def _build_prompt_static(self, source_of_truth: dict | None = None) -> str:
        return "\n\n".join([
            "あなたは強化学習の報酬設計エキスパートです。",
            self._source_of_truth_section(source_of_truth),
            self._titled_section("ゲーム概要",           self._prompt_section_game_overview()),
            self._titled_section("ゲームの目標",          self._prompt_section_game_objective()),
            self._titled_section("obs インデックス一覧",  self._prompt_section_obs_index()),
            self._titled_section("固定報酬（C++ 側）",    self._prompt_section_fixed_rewards()),
            self._titled_section("物理定数",              self._prompt_section_physics()),
            self._titled_section("武器タイプ別立ち回り指針", self._prompt_section_weapon_profiles()),
        ])

    def _build_prompt_dynamic(self, prev_metrics: dict | None, iteration: int,
                               prev_review: str | None = None,
                               initial_observation: str | None = None,
                               prev_reward_analysis: str | None = None) -> str:
        metrics_value = (
            "なし（初回）"
            if prev_metrics is None
            else json.dumps(prev_metrics, ensure_ascii=False, indent=2)
        )
        metrics_section = (
            f"## 前回イテレーション {iteration - 1} のメトリクス\n"
            f"### 各パラメーターの意味\n{self.metrics_description()}\n\n"
            f"### 値\n{metrics_value}"
        )
        task_section = (
            f"## タスク\n"
            f"以下のフォーマットで回答してください。\n\n"
            f"### 1. reward_fn.py のコード\n"
            f"```python\n"
            f"import numpy as np\n\n"
            f"def reward_shaping(obs: np.ndarray, prev_obs: np.ndarray, base_reward: float) -> float:\n"
            f'    """追加の報酬シェーピング関数。np.clip で [-1.0, 1.0] に収めること。"""\n'
            f"    shaped = 0.0\n"
            f"    return float(np.clip(shaped, -1.0, 1.0))\n"
            f"```\n\n"
            f"{self._titled_section('スケール制約', self._prompt_section_scale_constraints())}\n\n"
            f"### 2. 推奨訓練ステップ数\n"
            f"50,000〜200,000 の範囲で整数を記載してください\n\n"
            f"### 3. 設計の意図\n"
            f"この報酬関数で何を解決しようとしているか（1〜3行）"
        )
        items = [metrics_section]
        if initial_observation is not None:
            items.append(self._titled_section("ユーザーによる訓練観察・課題", initial_observation))
        if prev_reward_analysis is not None:
            items.append(self._titled_section(
                "前回イテレーションの報酬解析ログ（この情報を参考に reward_fn を改善してください）",
                prev_reward_analysis,
            ))
        if prev_review is not None:
            items.append(self._titled_section("前回レビューで指摘された設計上の問題", prev_review))
        items.append("## 課題\n前回の訓練の結果から課題を判断して箇条書きで記載。")
        items.append(task_section)
        return "\n\n".join(items)

    def build_prompt_parts(self, prev_metrics: dict | None, iteration: int,
                           prev_review: str | None = None,
                           initial_observation: str | None = None,
                           source_of_truth: dict | None = None,
                           prev_reward_analysis: str | None = None) -> tuple[str, str]:
        return self._build_prompt_static(source_of_truth), self._build_prompt_dynamic(
            prev_metrics, iteration, prev_review, initial_observation,
            prev_reward_analysis=prev_reward_analysis)

    def build_prompt(self, prev_metrics: dict | None, iteration: int,
                     prev_review: str | None = None,
                     initial_observation: str | None = None,
                     source_of_truth: dict | None = None,
                     prev_reward_analysis: str | None = None) -> str:
        static, dynamic = self.build_prompt_parts(prev_metrics, iteration, prev_review,
                                                   initial_observation, source_of_truth,
                                                   prev_reward_analysis=prev_reward_analysis)
        return static + "\n\n" + dynamic

    def build_constraints_hint(self, source_of_truth: dict | None = None) -> str:
        return "\n\n".join([
            self._source_of_truth_section(source_of_truth),
            self._titled_section("スケール制約", self._prompt_section_scale_constraints()),
            self._titled_section("物理定数",     self._prompt_section_physics()),
        ])

    # ------------------------------------------------------------------ #
    # メトリクス計算                                                        #
    # ------------------------------------------------------------------ #

    def compute_primary_metric(self, episode_base_rewards: list[float],
                               episode_lengths: list[int]) -> float:
        """primary_metric = アイテム+Kill スコアの平均（AliveReward 分を除去）。"""
        if not episode_base_rewards:
            return 0.0
        alive_reward = self._reward_constant("AliveReward", 0.001)
        scores = [
            max(0.0, r - alive_reward * l)
            for r, l in zip(episode_base_rewards, episode_lengths)
        ]
        return sum(scores) / len(scores)

    def compute_extra_metrics(self, episode_base_rewards: list[float],
                              episode_lengths: list[int]) -> dict:
        import statistics
        mean_len = sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
        alive_reward = self._reward_constant("AliveReward", 0.001)
        scores = [
            max(0.0, r - alive_reward * l)
            for r, l in zip(episode_base_rewards, episode_lengths)
        ]
        total_steps = sum(episode_lengths)
        total_score = sum(scores)
        return {
            "episode_length_mean": round(mean_len, 1),
            "episode_length_min":  min(episode_lengths) if episode_lengths else 0,
            "episode_length_max":  max(episode_lengths) if episode_lengths else 0,
            "item_kill_score_per_1k_steps": round(
                total_score / total_steps * 1000.0 if total_steps > 0 else 0.0, 3),
            "item_kill_score_std": round(
                statistics.stdev(scores) if len(scores) > 1 else 0.0, 3),
        }

    def metrics_description(self) -> str:
        alive_reward = self._reward_constant("AliveReward", 0.001)
        item_reward = self._reward_constant("ItemReward", 1.0)
        kill_reward = self._reward_constant("KillReward", 2.0)
        max_player_hp = self._player_constant("MaxPlayerHP", 100.0)
        return (
            f"- base_reward_mean: C++固定報酬の1エピソード平均（AliveReward={alive_reward}/step + ItemReward={item_reward} + KillReward={kill_reward}）\n"
            f"- shaped_reward_mean: reward_fn 出力の1エピソード平均（hp_penalty は含まない）\n"
            f"- hp_penalty_mean: 永続 HP ダメージペナルティの1エピソード平均\n"
            f"  計算式: clip(-hp_delta×100, -1, 0) / step。1HP ダメージ = -1.0 ペナルティ\n"
            f"  エピソード全体で最大 -{max_player_hp:.0f} になりうる（全 HP 消費時）\n"
            f"  ep_rew_mean(SB3) ~= base_reward_mean + shaped_reward_mean + hp_penalty_mean\n"
            f"- episode_reward_mean: base + shaped + hp_penalty の合計平均（SB3 の ep_rew_mean に対応）\n"
            f"- episode_length: エピソード長（ステップ数、最大 = 全 HP 消費まで）\n"
            f"- item_kill_score (primary): (base_reward - AliveReward×ep_len) の平均\n"
            f"  純粋なGem+Kill スコア。生存のみ=0.0、Gem1個={item_reward}、Kill1体={kill_reward}\n"
            f"- item_kill_score_per_1k_steps: 1000 step あたりのアイテム+Kill スコア。長く生きるだけでなく能動的に Gem/Kill を取れているかを見る効率指標\n"
            f"- item_kill_score_std: 標準偏差\n"
            f"- episode_length_min / max: 最短・最長エピソード長"
        )

    def make_model(self, env, device: str = "auto", weapon_phase: str = "W0"):
        """weapon_phase に対応した net_arch でモデルを生成する。"""
        from stable_baselines3 import PPO
        from common.utils import _linear_schedule
        from games.survivors.survivors_entity_attention_extractor import SurvivorsEntityAttentionExtractor
        from games.survivors.survivors_weapon_curriculum import WEAPON_PHASES

        phase_def = WEAPON_PHASES.get(weapon_phase, {})
        net_arch = phase_def.get("net_arch", [512, 256])  # デフォルト [512, 256]

        _obs_segments = (self._obs_schema or {}).get("segments", [])
        policy_kwargs = dict(
            features_extractor_class=SurvivorsEntityAttentionExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                offsets=self._offsets,
                obs_schema=_obs_segments,
            ),
            net_arch=net_arch,
        )
        print(f"[INFO] make_model: weapon_phase={weapon_phase}, net_arch={net_arch}")
        return PPO(
            "MlpPolicy", env,
            policy_kwargs=policy_kwargs,
            learning_rate=_linear_schedule(3e-4),
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
        )

    def make_curriculum_callback(
        self,
        raw_env,
        *,
        frame_skip: int,
        window: int,
        threshold_mult: float,
        alive_reward: float,
        status_path,
    ):
        if not hasattr(raw_env, "set_params"):
            return None
        from games.survivors.survivors_curriculum import CurriculumCallback
        return CurriculumCallback(
            raw_env=raw_env,
            frame_skip=frame_skip,
            window=window,
            threshold_mult=threshold_mult,
            alive_reward=alive_reward,
            status_path=status_path,
        )

    def collect_curriculum_metrics(self, curriculum_callback) -> dict:
        if curriculum_callback is None or not hasattr(curriculum_callback, "get_diagnostics"):
            return {}
        return curriculum_callback.get_diagnostics()

    def format_curriculum_summary(self, curriculum_metrics: dict) -> list[str]:
        rec = curriculum_metrics.get("recommendation", {})
        if not rec:
            return []
        return [
            "[Curriculum] 次回推奨値: "
            f"--curriculum-threshold {rec.get('suggested_curriculum_threshold')} "
            f"--curriculum-window {rec.get('suggested_curriculum_window')}",
            f"[Curriculum] threshold 根拠: {rec.get('threshold_reason')}",
            f"[Curriculum] window 根拠: {rec.get('window_reason')}",
        ]

    @property
    def primary_metric_name(self) -> str:
        return "item_kill_score"

    @property
    def default_port(self) -> int:
        return 8767


def create_config() -> SurvivorsEurekaConfig:
    return SurvivorsEurekaConfig()
