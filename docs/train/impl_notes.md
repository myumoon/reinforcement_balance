# 訓練実装ノート — 既知の問題・設計原則

過去の不具合修正・レビュー指摘から得られた教訓を集約したドキュメント。
新しいコールバック・Env・訓練スクリプトを実装するときに参照すること。

UE5 側の HTTP API 仕様・挙動は [`ue5_env.md`](ue5_env.md) を参照。

---

## 1. Python ↔ UE5 通信の設計原則

UE5 の `/params` はエピソード途中でも**即時反映**される（詳細は [`ue5_env.md`](ue5_env.md)）。
Python 側はこの挙動を前提に以下のルールを守ること。

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
class _SurvivorsMonitor(Monitor):
    def set_params(self, **kwargs) -> bool:
        return self.env.set_params(**kwargs)

    def set_shaping_weight(self, weight: float) -> None:
        self.env.set_shaping_weight(weight)
```

独自メソッドを持つ Env を Monitor でラップしたら、`env_method` で呼び出す全メソッドが
Monitor 経由で届くかを確認すること。

### パラメータ帰属のズレ（PR #126）

n_envs > 1 の場合、env0 が終了して新パラメータを全 env に送信すると、
env1〜3 はエピソード途中でパラメータが変わる。
そのまま env1 のエピソードが終わると「env1 が **新パラメータ** で走ったかのように ALP を計算」してしまう。

**解決策**: エピソード開始時点のパラメータを per-env で記録する。

```python
self._ep_start_param_vec_per_env: list[np.ndarray] = []

ep_param_vec = self._ep_start_param_vec_per_env[env_idx]  # 開始時点のparams
alp = self._compute_alp_for_episode(score_norm, ep_param_vec)
self._ep_start_param_vec_per_env[env_idx] = new_vec.copy()  # 終了したenvのみ更新
```

### Eval env と訓練 env のパラメータ同期（PR #116）

フェーズ型カリキュラムで eval env を生成した場合、現在のフェーズパラメータを
明示的に同期しないと eval が別条件で走る。

---

## 3. SB3 コールバック実装の注意点

### `training_env` は read-only

`BaseCallback.training_env` は `self.model.get_env()` を返す read-only property。
直接代入するとエラーになる。

### `env_method` のシグネチャ

```python
env_method(method_name, *method_args, indices=None, **method_kwargs)
```

- `indices=None`: 全 env に適用 / `indices=[0, 2]`: 指定した env のみ
- DummyVecEnv は**逐次実行**（並列ではない）

### `_on_training_start` での resume 上書きに注意（PR #125 レビュー）

```python
# 悪い例: import_state の _current_params が上書きされる
def _on_training_start(self):
    self._apply_params(_PHASE0_PARAMS)         # ← 復元値を破壊する
    self._current_params = dict(_PHASE0_PARAMS)

# 良い例
def _on_training_start(self):
    self._apply_params(self._current_params)   # resume 時は復元値、新規は PHASE0_PARAMS
```

### GMM fitting の `break` vs `continue`（PR #125 レビュー）

`except: break` だとそれ以降の k が全て試されなくなる。`continue` を使うこと。

---

## 4. スコア正規化（SPALF 固有）

### `score_norm > 1.0` になると ALP が破綻する

`score_norm = active_score / spalf_max_score > 1.0` になると全領域の ALP が一様に膨らみ、
GMM が有意な高 ALP 領域を見つけられなくなる。
`score_norm > 1.0` のエピソードが 5% を超えたら `spalf_max_score` を見直すこと。

### `spalf_max_score` の設定基準

`PHASE0_PARAMS` は非常に簡単な設定のためスコアがフェーズ型の目標値より大幅に膨らむ。
最初の数百エピソードの `active_score` 最大値を観察し、その 2〜3 倍を `spalf_max_score` に設定する。

---

## 5. W&B / ログ設計

### W&B ログに `global_step` を必ず付けること（PR #106）

```python
wandb.log(metrics, step=self.num_timesteps)  # step= なしだと X 軸が自動採番になる
```

### 複数ソースからの run_id 読み込みは排他処理にすること（PR #125 レビュー）

`spalf_state.json` と `wandb_run_id.txt` の両方を `if` で読むと後者で上書きされる。
`if/elif` で排他処理にすること。

### kwargs が config 設定を上書きする問題

W&B config の値（`ent_coef` など）が `_PPO_KWARGS` に同名キーで存在すると
resume 時に上書きされる。複数箇所で同じキーを設定する場合は優先順位を明確にすること。

---

## 6. Resume 実装

### `run_dir` の誤検出（PR #105, #111）

`--resume` にファイルパスを渡すと `run_dir` がモデルファイルの親ディレクトリになる。
`log/train_status.json` が存在するディレクトリを遡って解決する処理が必要。

### `--resume + --config` の組み合わせ（PR #111）

`config` ファイルの親ディレクトリを `run_dir` として自動推論する処理が必要。

### 変数名の混在（PR #112）

`resume_source_dir` と `source_dir` など類似変数名の混在は `NameError` の原因になる。
resume 関連変数は命名を統一すること。
