# reinbalance-survivors-contracts

Survivors の Training と Deployment が、実行ディレクトリや `sys.path` hack に依存せず
同じ契約型・schema hash・UI policy を共有するための installable package。

## 提供するもの

- `canonical_json` — UTF-8 / sorted keys / finite-only の canonical JSON bytes と SHA-256。
- `UiIntentV1` — 全 UI 決定の唯一の共有表現（kind ごとの one-of validation、effect owner 固定）。
- `UiPolicyInputV1` / `NonModelUiPolicyConfigV1` / `decide_non_model_ui_intent()` —
  fallback / meta / ack / confirm の pure decision rule（`NonModelUiPolicyV1`）。
- `DeployObsSchema` / `DeployObservation`、`ItemDecisionFeatures` / `CandidateFeatures`、
  `PerceptionErrorProfile`、`TargetProfileRef` / `ActionSemantics` の versioned wire 型。

## インストール

```bash
python -m pip install -e Tools/Common
```

## 依存境界

このパッケージは SB3 / PyTorch / UE5 HTTP client / OpenCV / DXcam / Training・Deployment
module を import してはならない。NumPy は lazy array helper でのみ使用する。

依存方向は `Common <- Training`, `Common <- Deployment`。逆方向・相互 import は禁止。
