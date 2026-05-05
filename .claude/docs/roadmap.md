# 今後の実装プラン

## ロードマップ現状

| フェーズ | 項目 | 状態 |
|---------|------|------|
| Phase 1-A | HP/ダメージシステム | ✅ 完了（SurvivorsGame） |
| Phase 1-A | Aura 攻撃（動的半径） | ✅ 完了（MinAuraRadius〜MaxAuraRadius lerp） |
| Phase 1-A | XP・レベルシステム | ✅ 完了（SurvivorsGame） |
| Phase 1-A | フレームスキップ（Action Repeat） | ✅ 完了（PR #32） |
| Phase 1-B | UE5 `/params` エンドポイント | ✅ 完了（SurvivorsGame） |
| Phase 1-B | **Python CurriculumCallback** | ❌ **未実装 ← 次の作業** |
| Phase 2 | W&B ログ整備 | ❌ 未実装 |
| Phase 2 | 並列環境対応 | △ 部分（UE5 単一接続制約で n_envs=1） |
| Phase 3 | LSTM / Hierarchical RL 等 | ❌ 未実装（将来） |

---

## Phase 1-B: Python CurriculumCallback

### 背景
UE5 の `/params` エンドポイント（MaxActiveEnemies / EnemySpeedMult / SpawnInterval）は実装済み。
`SurvivorsEnv.set_params()` も実装済み。Python 側の自動難易度制御が未接続。

### 実装内容

#### `Tools/Training/train.py` に追加
```python
class CurriculumCallback(BaseCallback):
    """直近 window エピソードの item_kill_score 平均が threshold を超えたら難易度を上げる。"""
    def __init__(self, env, window=20, threshold=5.0, verbose=0):
        super().__init__(verbose)
        self.env = env       # SurvivorsEnv（set_params を持つ）
        self.window = window
        self.threshold = threshold
        self._scores: list[float] = []
        self._stage = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                # item_kill_score = base_reward - AliveReward * ep_len
                base = info.get("base_reward", 0.0)
                ep_len = info["episode"]["l"]
                self._scores.append(max(0.0, base - 0.001 * ep_len))
                if len(self._scores) >= self.window:
                    mean = sum(self._scores[-self.window:]) / self.window
                    if mean >= self.threshold:
                        self._advance_stage()
        return True

    def _advance_stage(self):
        self._stage += 1
        self._scores.clear()
        enemies = min(6 + self._stage, 20)
        speed   = min(1.0 + self._stage * 0.2, 3.0)
        spawn   = max(5.0 - self._stage * 0.5, 2.0)
        self.env.set_params(
            MaxActiveEnemies=enemies,
            EnemySpeedMult=speed,
            SpawnInterval=spawn,
        )
        print(f"[Curriculum] Stage {self._stage} — 敵数={enemies}, 速度×{speed:.1f}, スポーン{spawn:.1f}s")
```

CLI 引数追加:
```python
p.add_argument("--curriculum", action="store_true",
               help="カリキュラム学習（SurvivorsGame のみ）")
p.add_argument("--curriculum-window", type=int, default=20,
               help="評価ウィンドウサイズ（エピソード数）")
p.add_argument("--curriculum-threshold", type=float, default=5.0,
               help="次ステージに進む item_kill_score 閾値")
```

#### `Tools/Training/eureka_loop.py` にも同様に適用（オプション引数）

### 影響ファイル
- `Tools/Training/train.py`
- `Tools/Training/eureka_loop.py`

### 検証
```bash
python train.py --game survivors --curriculum --total-steps 300000
# → "[Curriculum] Stage X に進みました" がログに出ること
# → UE5 側の敵数・速度が段階的に増えること
```

---

## Phase 2-A: W&B ログ整備

### 背景
現在のログは TensorBoard（SB3 デフォルト）と print のみ。
実験管理・ハイパーパラメータ比較・カリキュラム進行の可視化に W&B が必要。

### 実装内容

#### `Tools/Training/requirements.txt` に追加
```
wandb>=0.17.0
```

#### `Tools/Training/train.py` に統合
```python
import wandb
from wandb.integration.sb3 import WandbCallback
```

起動時の init:
```python
if args.wandb:
    wandb.init(
        project="rl-balance",
        name=args.output.stem if args.output else None,
        config={
            "game": args.game,
            "frame_skip": args.frame_skip,
            "total_steps": args.total_steps,
            "curriculum": args.curriculum,
            **ppo_kwargs,
        },
        sync_tensorboard=True,
    )
    callbacks.append(WandbCallback(verbose=0))
```

カリキュラム進行のログ（`CurriculumCallback._advance_stage` 内）:
```python
if wandb.run:
    wandb.log({"curriculum/stage": self._stage,
               "curriculum/max_enemies": enemies,
               "curriculum/speed_mult": speed})
```

CLI 引数:
```python
p.add_argument("--wandb", action="store_true", help="W&B ログを有効にする")
```

#### `Tools/Training/eureka_loop.py` にも同様に適用（イテレーション・報酬関数ごとに run を分けるか group 化）

### 影響ファイル
- `Tools/Training/requirements.txt`
- `Tools/Training/train.py`
- `Tools/Training/eureka_loop.py`

---

## Phase 2-B: 並列環境対応

### 背景
現在 n_envs=1 に制限（UE5 HTTP サーバーが 1 ポートに固定）。
複数 UE5 スタンドアロンインスタンスを起動することで n_envs=N を実現できる。

### 制約
- UE5 エディタ PIE は 1 インスタンスのみ → **パッケージビルドが前提**
- 各インスタンスに異なるポートを割り当てる必要がある

### 設計方針

#### UE5 側の変更
- 起動引数でポート番号を受け取れるように `HttpEnvServerBase` のポートをコマンドライン引数化
  - 例: `SurvivorsStandaloneGame.exe -port=8767`
  - UE5 の `FCommandLine::Get()` で引数解析 → `BeginPlay` でポート設定

#### Python 側の変更 (`Tools/Training/train.py`)

```python
p.add_argument("--num-envs", type=int, default=1,
               help="並列環境数（パッケージ版 UE5 が必要）")
p.add_argument("--base-port", type=int, default=None,
               help="並列時のベースポート（--num-envs 個分のポートを確保）")
```

複数ポートでの VecEnv 構築:
```python
if args.num_envs > 1:
    ports = [base_port + i for i in range(args.num_envs)]
    def _make_env_for_port(p):
        def _inner():
            e = SurvivorsEnv(host=args.host, port=p, frame_skip=args.frame_skip)
            e._reward_fn = reward_fn
            return e
        return _inner
    env = SubprocVecEnv([_make_env_for_port(p) for p in ports])
```

### 影響ファイル
- `ReinBalance/Source/ReinBalance/Private/SurvivorsGame.cpp`（コマンドライン引数によるポート変更）
- `ReinBalance/Source/ReinBalanceEditor/Private/Training/SurvivorsHttpEnvService.cpp`（同上）
- `Tools/Training/train.py`

### 備考
パッケージビルドが必要なため、Phase 2-A（W&B）より後に実施。
学習速度は frame_skip で代替可能なため優先度は低め。

---

## Phase 3（将来）

- **LSTM Policy**: 部分観測問題への対応（敵の見えない方向を記憶）
- **Hierarchical RL**: 「アイテム取得」「敵回避」の階層的目標分解
- **World Model (Dreamer)**: 環境モデルを学習して想像上でのロールアウト
