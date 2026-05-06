# coin_count obs 追加（累積コイン収集数）

**実装状況: ❌ 却下・不採用（2026-05-06）**

## 却下理由

`xp_progress` と `player_level` の obs が既に存在しており、これらからコイン収集を間接的に検出できるため不要と判断された。

- `obs[xp_progress_idx] > prev_obs[xp_progress_idx]` でコイン収集（XP増加）を検出できる
- `player_level` もコイン収集に連動するため、状態は十分に観測可能

実装コストに対して得られる情報は既存 obs と重複するため、obs 次元を増やさないことを優先した。

---

## 以下は没になった設計（参考記録）

### Context

reward_fn.py でコイン収集イベントを検出するには現在 `base_reward >= 4.0` という
浮動小数点の閾値比較に頼っているが、より確実な方法としてエピソード内の
累積コイン収集数を obs に直接含める案があった。

**重要な前提**: `CoinPositions.Num()` はコインが「再配置」されるため常に `NumCoins` で一定。
「フィールド上のコイン数」ではなく **エピソード内累積収集数** (`CoinsCollected`) を追加する案だった。

検出パターン（没案）: `obs[coins_collected_idx] > prev_obs[coins_collected_idx]` → このステップでコインを収集した

### 変更予定だったファイル

- `ReinBalance/Source/ReinBalance/Public/CoinGame.h` — `CoinsCollected` メンバー追加、`GetObsDim()` +1
- `ReinBalance/Source/ReinBalance/Private/CoinGame.cpp` — obs 追加、リセット時クリア
- `Tools/Training/games/coin/coin_eureka_config.py` — obs インデックス説明追加
