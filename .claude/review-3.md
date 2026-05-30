# コードレビュー — 2026-05-31 08:22

**ブランチ:** feature-a-20260531015501 vs main
**タスク説明:** 訓練基盤リファクタリング（プラン: 20260531005314_training-refactor.md）

## 総評

責務分離の方向性は妥当で、ParamApplier / WandbLogger / StateModule への抽出により訓練基盤の見通しは良くなっています。一方で、SPALF の resume 時に復元済みパラメータを捨てて Phase 0 を適用する重大な挙動変更があります。また、CurriculumCallback から episode base/alive reward と terminated/truncated の記録が抜け、診断・保存状態の一部が常に空または 0 になります。pytest / py_compile は `python` shim が WSL から実行できず、起動前に失敗しました。

## 指摘事項

### ❌ Tools/Training/games/survivors/spalf_callback.py:46 — SPALF resume が常に Phase 0 から再開される

`import_state()` は `SpalfStateModule` に `_current_params` を復元しますが、`_on_training_start()` が無条件に `dict(_PHASE0_PARAMS)` を作って env に適用し、`_spalf._current_params` / `_current_param_vec` も上書きしています。旧実装は resume 時に復元済みの `_current_params` を適用していたため、学習再開時に探索中の難易度が失われ、SPALF の状態と実際の env パラメータが Phase 0 に巻き戻ります。`initial_params = dict(self._spalf._current_params)` を使い、復元済み状態がある場合はそれを env に適用してください。

[ファイルを開く](vscode://file//mnt/d/prog/0_myprogram/ue5_reinforcement_balance/Tools/Training/games/survivors/spalf_callback.py:46:1)

### ⚠️ Tools/Training/games/survivors/state_modules.py:355 — Curriculum の診断用 episode 統計が更新されなくなっている

`CurriculumStateModule.on_episode_end()` は active_score と ep_len だけを受け取るため、旧 `CurriculumCallback` が記録していた `episode_base_rewards`、`episode_alive_rewards`、`terminated_count`、`truncated_count`、`missing_episode_info_count` が更新されません。その結果、`get_diagnostics()` / `get_wandb_progress_metrics()` の base/alive reward 平均や terminated/truncated ratio が空配列由来の 0 になり、resume 用 JSON も実態を失います。`EpisodeScoreTracker` の戻り値に ep_base / alive_total / termination 情報を含めるか、CurriculumCallback 側で info を渡して state module が旧実装と同じ統計を更新できる形に戻してください。

[ファイルを開く](vscode://file//mnt/d/prog/0_myprogram/ue5_reinforcement_balance/Tools/Training/games/survivors/state_modules.py:355:1)

### 💡 Tools/Training/games/survivors/state_modules.py:567 — CurriculumStateModule のメソッド群が重複定義されている

`CurriculumStateModule` 内で `__init__` から `save_status()` までが同じクラス内にもう一度定義されています。Python では後勝ちになるため現状の実行結果は大きく変わりませんが、片側だけ修正した場合に反映されない、レビュー時に実際に使われる定義を誤認する、といった保守リスクがあります。重複ブロックを削除して定義を一箇所にしてください。

[ファイルを開く](vscode://file//mnt/d/prog/0_myprogram/ue5_reinforcement_balance/Tools/Training/games/survivors/state_modules.py:567:1)

## 良い点

- UE5 の `set_params` 呼び出しを `ParamApplier` に集約したことで、Curriculum / SPALF / 将来の Hybrid 実装で同じ適用経路を使えるようになっています。
- W&B への直接依存を `WandbLogger` 注入に寄せたことで、コールバック単体のテスト容易性とロガー差し替え余地が改善されています。
- `EpisodeScoreTracker` と state module の抽出により、スコア計算・状態保存・遷移判定を個別にテストしやすい構造になっています。

---

アイコン凡例: `❌` バグ・重大な問題 / `⚠️` 修正推奨 / `💡` 改善提案
