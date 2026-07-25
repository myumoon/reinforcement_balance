# DeployObsV1

DeployObsV1 は simulator と将来の real screen parser が共有する、画面観測だけで生成可能な policy 入力契約です。既存の privileged raw observation は teacher と diagnostics 用に維持し、release policy の入力には使いません。

各 feature は同じ長さの `values`、`validity`、`age` を持ち、policy tensor はこの順に連結します。欠損・画面外・観測不能は schema の neutral 値、validity `0`、age `1` です。NaN は使用しません。age は `min(age_ms / max_age_ms, 1)`、stale threshold 後の validity は max age に向けて線形に減衰します。

world feature は viewport 中心基準の `[-1,1]` screen-space です。count は visible かつ非 occluded・非 clipped の track だけを数えます。screen-to-world unit 変換、画面外座標、hidden HP/cooldown、全 state の count/density は release 契約ではありません。categorical id は vocabulary 末尾に `unknown` を予約して `[0,1]` に正規化します。

schema layout は `Tools/Deployment/configs/deploy_obs_v1.yaml` の segment 名と記載順から生成され、絶対 offset を設定しません。schema hash は共有 canonical JSON 実装だけで計算します。real parser は未実装ですが、synthetic named estimates が同じ adapter と schema hash/dim/range gate を通るため、将来の parser もこの境界へ接続します。

Training の `DeployObsWrapper.release()` は camera projection、visibility/occlusion/clipping、named estimates の順で変換します。`oracle_diagnostic()` は比較診断専用で、release artifact の生成を gate で禁止します。VecNormalize は deploy tensor の外側へ新規 fit し、privileged source の統計を流用しません。

DeployObs schema または release adapter の producer hash が変わると、既存 00-05 fidelity baseline は意図的に失効します。`survivors.sim_real_fidelity.v2`、13 gating key、`fidelity_producer_paths` allowlist、`verify_current_fidelity` の意味は変更しません。01-05 formal 収集前に integration fidelity verdict を再発行してください。
