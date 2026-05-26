# 訓練実装ノート — 既知の問題・設計原則

過去の不具合修正・レビュー指摘から得られた教訓を集約したドキュメント。
新しいコールバック・Env・訓練スクリプトを実装するときに参照すること。

---

## 1. UE5 ↔ Python 通信の設計原則

### `/params` エンドポイントは即時反映される

`/params` を受信した UE5 は `Game->MinActiveEnemies = ...` のように**その場で直接代入**する。
エピソードのリセットや次ステップまで待機する処理は一切ない。

```
Python が /params を送信
  → UE5 が即座に Game->MinActiveEnemies, EnemySpeedMult ... を上書き
  → 以降のステップは新しいパラメータで動き続ける（エピソード継続中でも）
```

エピソード途中に送信すると**前半と後半で難易度が変わる**。
これは ALP 計算の前提（「このパラメータでこのスコアが出た」）を崩す原因になる。

### 送信先は必要最低限の env に絞ること（PR #127）

UE5 の 4 インスタンスはそれぞれ別ポートで動作しており、相互通信はしない。
`env_method("set_params", ...)` はデフォルトで全 env に送信するため、
不要な env への送信を避けるには `indices` 引数を使う。

```python
# 悪い例: 全 env に送信（途中エピソードの env にも届く）
self.training_env.env_method("set_params", **ue5_params)

# 良い例: 終了した env にのみ送信
self.training_env.env_method("set_params", indices=[env_idx], **ue5_params)
```

`_on_training_start` など全 env を初期化したい場面では `indices=None`（デフォルト）のままでよい。

### HTTP 通信コストを見積もること

n_envs=4、平均エピソード長 300 ステップの場合:

| パターン | 1 ロールアウトあたりの HTTP 送信回数 |
|---|---|
| 全 env に毎エピソード送信（旧設計） | ~56 回 |
| 終了した env のみに送信（現設計） | ~14 回 |
| フェーズ昇格時のみ送信（フェーズ型カリキュラム） | ~0〜1 回 |

HTTP 送信回数は訓練速度に直結する。不要な送信はパフォーマンスを 2 倍程度低下させることがある。

### ゲームパラメータ（Survivors / `/params`）

各パラメータの定義・デフォルト値・説明コメントは以下のファイルを Source of Truth とすること:

```
ReinBalance/Source/ReinBalance/Public/Survivors/Logic/SurvivorsGame.h
```

`/params` エンドポイントでの受信・適用処理は以下を参照:

```
ReinBalance/Source/ReinBalanceEditor/Private/Training/SurvivorsHttpEnvService.cpp
```

---

## 2. 多並列環境（n_envs > 1）特有の問題

### env は必ず Monitor でラップすること（PR #113）

SB3 は `Monitor` が収集する `ep_rew_mean` / `ep_len_mean` などの統計情報を使って
ログの **rollout セクション**を出力する。`Monitor` でラップしないと rollout セクションが
空のまま出力されず、訓練の進捗をログから確認できなくなる。

```python
# 悪い例: rollout セクションが出力されない
vec_env = DummyVecEnv([lambda: SurvivorsUE5Env(port=port)])

# 良い例: Monitor でラップしてから VecEnv に渡す
from stable_baselines3.common.monitor import Monitor
vec_env = DummyVecEnv([lambda: Monitor(SurvivorsUE5Env(port=port))])
```

### Monitor でラップした場合は独自メソッドを明示フォワードすること（PR #114）

`Monitor` は `gym.Wrapper` を継承しているが、**独自メソッド**（`set_params` / `set_shaping_weight` など）
は自動的に透過しない。VecEnv の `env_method` からこれらを呼び出すと `AttributeError` になる。

```python
# _SurvivorsMonitor で明示的にフォワードする
class _SurvivorsMonitor(Monitor):
    def set_params(self, **kwargs) -> bool:
        return self.env.set_params(**kwargs)

    def set_shaping_weight(self, weight: float) -> None:
        self.env.set_shaping_weight(weight)
```

**教訓**: 独自メソッドを持つ Env を Monitor でラップしたら、その後 `env_method` で呼び出す
全てのメソッドが Monitor 経由で届くかを確認すること。

### パラメータ帰属のズレ（PR #126）

n_envs > 1 の場合、env0 が終了して新パラメータを全 env に送信すると、
env1〜3 はエピソード途中でパラメータが変わる。
そのまま env1 のエピソードが終わると「env1 が **新パラメータ** で走ったかのように ALP を計算」してしまう。

```
env0 終了 → params_A を全 env に送信
env1（途中）が params_A を受け取る
env1 終了 → ALP を params_A で計算 ← 実際は params_B で走っていたのに！
```

**解決策**: エピソード開始時点のパラメータを per-env で記録する。

```python
self._ep_start_param_vec_per_env: list[np.ndarray] = []

# エピソード終了時は ep_start_param_vec_per_env[env_idx] を参照
ep_param_vec = self._ep_start_param_vec_per_env[env_idx]
alp = self._compute_alp_for_episode(score_norm, ep_param_vec)

# 次エピソードの開始パラメータを更新するのは終了した env のみ
self._ep_start_param_vec_per_env[env_idx] = new_vec.copy()
```

### Eval env と訓練 env のパラメータ同期（PR #116）

eval env はフェーズ型カリキュラムでパラメータを動的に変更する場合、
訓練 env と同じフェーズパラメータを明示的に同期しないと、eval が別条件で走る。
eval env 生成後に現在のフェーズパラメータを送信する処理が必要。

---

## 3. SB3 コールバック実装の注意点

### `training_env` は read-only

`BaseCallback.training_env` は `self.model.get_env()` を返す read-only property。
直接代入しようとするとエラーになる。

```python
self.training_env = vec_env  # ← エラー（read-only）
```

### `env_method` のシグネチャ

```python
env_method(method_name, *method_args, indices=None, **method_kwargs)
```

- `indices=None`: 全 env に適用
- `indices=[0, 2]`: 指定した env のみに適用
- DummyVecEnv は**逐次実行**（並列ではない）

### `_on_training_start` での resume 上書きに注意（PR #125 レビュー）

`import_state()` でロード済みの状態を `_on_training_start` で上書きしてしまうパターン:

```python
# 悪い例: resume 時に import_state の _current_params が上書きされる
def _on_training_start(self):
    self._apply_params(_PHASE0_PARAMS)    # ← import_state の内容を破壊する
    self._current_params = dict(_PHASE0_PARAMS)  # ← 同上

# 良い例: 既に復元済みの _current_params をそのまま使う
def _on_training_start(self):
    self._apply_params(self._current_params)  # resume 時は復元値、新規は PHASE0_PARAMS
```

### GMM fitting の `break` vs `continue`（PR #125 レビュー）

k を増やしながら GMM をフィットするループで `except: break` とすると
**それ以降の k が全て試されなくなる**。

```python
# 悪い例: k=3 で失敗したら k=4〜10 が試されない
for k in range(1, K_MAX + 1):
    try:
        ...
    except Exception:
        break  # ← 以降の k がスキップされる

# 良い例: その k だけスキップして次を試す
for k in range(1, K_MAX + 1):
    try:
        ...
    except Exception:
        continue
```

---

## 4. スコア正規化（SPALF 固有）

### `score_norm > 1.0` になると ALP が破綻する

`score_norm = active_score / spalf_max_score` が 1.0 を超えると:

1. 全パラメータ領域の ALP が一様に大きくなる
2. GMM が「どの領域で本当に学習が進んでいるか」を判別できなくなる
3. パラメータが意味なく振動する

`score_norm > 1.0` のエピソードが全体の 5% を超えたら `spalf_max_score` の見直しを検討する。

### `spalf_max_score` の設定基準

`PHASE0_PARAMS`（最も簡単な設定）は `enemy_hp_scale=0.5` / `enemy_damage_scale=0.5` /
`time_scaling=False` など非常に易しく、スコアがフェーズ型の目標値より大幅に膨らむ。

```
悪い例: spalf_max_score = 2250  （Phase 15 クリア閾値）
  → PHASE0_PARAMS での実績スコア 13,000+ → score_norm ≈ 5.8 → ALP 破綻

良い例: spalf_max_score = 30000  （実績最大スコアより十分に大きい値）
  → score_norm が [0, 1] 程度に収まる
```

**設定基準**: 最初の数百エピソードの `active_score` の最大値を観察し、
その 2〜3 倍の値を `spalf_max_score` として設定する。

---

## 5. W&B / ログ設計

### W&B ログに `global_step` を必ず付けること（PR #106）

`wandb.log(metrics)` に `step=` を指定しないと W&B の X 軸が自動採番になり、
複数 run の比較ができなくなる。

```python
wandb.log(metrics, step=self.num_timesteps)  # ← 必ず step= を指定
```

### kwargs が config 設定を上書きする問題（PR #125 付近）

W&B の config として保存された値（`ent_coef` など）が `_PPO_KWARGS` に
同名キーとして存在すると、resume 時に後から読んだ方で上書きされる。
同じキーが複数の場所で設定される場合は読み込み順序と優先順位を明確にすること。

### 複数ソースからの run_id 読み込みは排他処理にすること（PR #125 レビュー）

SPALF の `spalf_state.json` と通常の `wandb_run_id.txt` の両方から run_id を
読もうとすると、後から読んだ方で上書きされる。

```python
# 悪い例: 両方を読んで後者が上書き
if spalf_state_exists:
    wandb_run_id = spalf_state["wandb_run_id"]
if wandb_id_path.exists():
    wandb_run_id = wandb_id_path.read_text()  # ← spalf_state の値が消える

# 良い例: 排他 (if/elif)
if args.spalf and spalf_state_exists:
    wandb_run_id = spalf_state["wandb_run_id"]
elif wandb_id_path.exists():
    wandb_run_id = wandb_id_path.read_text()
```

---

## 6. Resume 実装

### `--resume` にファイルパスを渡したときの `run_dir` 誤検出（PR #105, #111）

`--resume runs/survivors/v06/train/run/work/model_steps/model_1000000_steps.zip` のように
ファイルパスを渡すと、`run_dir` がモデルファイルの親ディレクトリ（`model_steps/`）に
なってしまう問題。`run_dir` は `log/train_status.json` が存在するディレクトリとして
遡って解決する処理が必要。

### `--resume + --config` の組み合わせ（PR #111）

`--resume` と `--config` を同時に指定した場合、`config` ファイルの親ディレクトリを
`run_dir` として自動推論する処理が必要。明示的な `run_dir` 指定がない場合のフォールバック順序を設計しておくこと。

### 変数名の混在に注意（PR #112）

`resume_source_dir` と `source_dir` のような類似変数名が混在すると
参照ミスで `NameError` になる。resume 関連変数は命名を統一すること。
