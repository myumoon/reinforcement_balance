# DeployObsV1

DeployObsV1 は simulator と将来の real screen parser が共有する、画面観測だけで生成可能な policy 入力契約です。既存の privileged raw observation は teacher と diagnostics 用に維持し、release policy の入力には使いません。

各 feature は同じ長さの `values`、`validity`、`age` を持ち、policy tensor はこの順に連結します。欠損・画面外・観測不能は schema の neutral 値、validity `0`、age `1` です。NaN は使用しません。age は `min(age_ms / max_age_ms, 1)`、stale threshold 後の validity は max age に向けて線形に減衰します。

world feature は viewport 中心基準の `[-1,1]` screen-space です。count は visible かつ非 occluded・非 clipped の track だけを数えます。screen-to-world unit 変換、画面外座標、hidden HP/cooldown、全 state の count/density は release 契約ではありません。categorical id は vocabulary 末尾に `unknown` を予約して `[0,1]` に正規化します。

schema layout は `Tools/Deployment/configs/deploy_obs_v1.yaml` の segment 名と記載順から生成され、絶対 offset を設定しません。schema hash は共有 canonical JSON 実装だけで計算します。real parser は未実装ですが、synthetic named estimates が同じ adapter と schema hash/dim/range gate を通るため、将来の parser もこの境界へ接続します。

Training の `DeployObsWrapper.release()` は camera projection、visibility/occlusion/clipping、named estimates の順で変換します。`oracle_diagnostic()` は比較診断専用で、release artifact の生成を gate で禁止します。VecNormalize は deploy tensor の外側へ新規 fit し、privileged source の統計を流用しません。

DeployObs schema または release adapter の producer hash が変わると、既存 00-05 fidelity baseline は意図的に失効します。`survivors.sim_real_fidelity.v2`、13 gating key、`fidelity_producer_paths` allowlist、`verify_current_fidelity` の意味は変更しません。01-05 formal 収集前に integration fidelity verdict を再発行してください。

## DeployObs v2

DeployObs v2（`deploy_obs.v2`）は、v1 の10 segment を先頭に同じ名前・設定で残し、敵・全ジェム・レアジェム（緑・赤）の16方向特徴（最寄り距離・近距離密度・中距離密度）、武器・パッシブのスロット種類とレベル、武器エフェクト（aura 半径・orbit 周回半径・最寄り4個の zone・projectile 方向別密度）とそのスロット・残り時間を後ろへ足した契約です。value 面は222次元（3面で666次元）です。既定 schema は v1 のままで、Training は 03-07・03-08、Deployment は 04-13 で v2 へ切り替えます。

- 置き場所: schema は `Tools/Common/src/reinbalance_survivors_contracts/deploy_obs.py` の `DeployObsSchema.default_v2()` と package-data の `schemas/deploy_obs_v2.yaml`（両者の一致をテストで固定）。特徴量ビルダーは `deploy_obs_v2_features.py` の `build_deploy_obs_v2()`、距離帯・語彙・武器→エフェクト種類・持続時間表・weapon 4クラスの `max_age` フレーム数は `schemas/deploy_obs_v2_features.yaml` にだけ置きます。Training と Deployment は特徴量を自分で計算せず、この関数を呼びます。
- 座標系: 位置・距離・半径はすべてプレイヤー基準・縦横同じ縮尺・viewport 半幅で正規化し（`dx = (x_px - player_x_px) / (W/2)`、`dy` も `W/2` で割る）、出力する位置の値は `[-1,1]` に clip します。方向ビン・距離帯・zone の並び順は clip 前の値で求めます（先に clip すると画面端のプレイヤーから見た方向が歪むため）。`player_screen_pos` は `(player_px - center_px) / (W/2)` です。v1 にあった Training（`2x/w-1`）と実機（`[0,1]` 画像座標の差分）のスケール不一致は v2 に持ち込みません。
- 方向ビン・距離帯: 方向ビンは C++ `BuildDirDensity` と同じ `floor((atan2(dy,dx)+π)/2π × 16)` で、距離が `KINDA_SMALL_NUMBER` 以下の点は除外します。近距離・中距離帯の境界と密度の正規化係数は画面内に収まる値を yaml に記録し、教師の 600u / 1400u は流用しません。
- 可視規則: track は中心が viewport 内にあり遮蔽されていなければ可視です。矩形が画面外へはみ出していても落とさず、実機 tracker の `clipped` フラグは v2 の可視判定に使いません。
- 確定した不在と不明の区別:
  - 画面全体を走査して作る特徴（16方向特徴、projectile 密度、zone の空き枠、最寄りの敵、敵数）は、見えなければ「画面内に無い」という観測として neutral、validity は world 認識全体の有効性に従います。
  - aura・orbit の半径は、その種類を出す武器を HUD で持っていないと確定すれば neutral・validity 1、持っているのに見えなければ neutral・validity 0・age 1 です。HUD のスロットが読めていない場合も不明として validity 0 にします。
  - aura・orbit・zone のスロットと残り時間は、その種類を出しうる武器を HUD 上でちょうど1つ持つと確定したときだけ有効です（Fire Wand と Santa Water を同時に持つと zone は validity 0）。残り時間は `持続時間(武器, レベル, 持続時間倍率) − (現在時刻 − 初観測時刻)` を 8 秒で割った値で、レベルや持続時間倍率が不明なら validity 0 です。出しうる武器が無いと確定すれば neutral・validity 1 です。
    - 既知の制約: スロット番号 `slot / 5` はスロット0で 0.0 になり neutral と同じ値です。このため slot 面だけでは「スロット0の武器が出している」と「出しうる武器が無いと確定」を区別できません（`weapon_slot_ids` と合わせれば区別できます）。neutral を変えるかどうかは 03-05 の学習前に plan 側で判断します。
  - validity 0 の要素は必ず neutral・age 1、validity 1 の要素は age 0 です。NaN は使いません。
- C++ との一致: 武器・パッシブの語彙（`EWeaponType` / `EPassiveItemType`）、上限定数、持続時間表、武器→エフェクト種類の集合は `Tools/Common/tests/test_deploy_obs_v2_cpp_parity.py` が C++ ソースを読んで一致を確認します。
- golden fixture: `Tools/Common/tests/fixtures/deploy_obs_v2_golden_v1.json` は px 入力と期待 tensor の組です。03-07（sim 経路）と 04-13（実機経路）は、自分の入力変換を通した結果がこの期待値と一致することをテストします。
- fidelity: `fidelity_producer_paths_v1.json` の `deploy_obs_schema` と `deploy_release_adapter` に上記の Python モジュールと yaml を登録しており、内容が変わると gating hash が変わって古い判定を stale として検出できます。
