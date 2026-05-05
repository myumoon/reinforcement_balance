# UE5 強化学習プラグイン・訓練方法の比較

作成日: 2026-03-xx  
更新日: 2026-05-06（採用アーキテクチャの記述を実態に合わせて修正）

## 概要

UE5で強化学習を行う場合、主に3つのアプローチがあります。  
本ドキュメントでは各アプローチの特徴・メリット・デメリット・並列訓練対応状況を整理します。

---

## アーキテクチャ比較

### 1. カスタム HTTP（採用）

```
Python (SB3 PPO)
  ↕ HTTP REST API (localhost:port)
UE5 Editor (PIE) — GameThread で物理ステップ実行
  ├── POST /reset   → エピソードリセット → obs 返却
  ├── POST /step    → アクション送信 → obs/reward/done 返却
  ├── GET  /obs_schema → 観測定義（起動時1回）
  └── POST /params  → 難易度パラメータ動的変更
         ↓（訓練完了後）
  ONNX エクスポート → NNERuntimeORT で推論（C++ のみで完結）
```

**メリット**
- RLフレームワーク（SB3, RLlib, CleanRL等）を自由に選択可能
- 観測データの形式・通信頻度を完全に制御できる
- UE5 プラグイン不要。HTTPServer モジュールのみ
- frame_skip（Action Repeat）を UE5 側のN-ループで効率化できる

**デメリット**
- 並列化は複数 UE5 インスタンスが必要（PIE は1インスタンスのみ）
- 通信レイテンシがボトルネックになりうる

**並列訓練**  
`SurvivorsEnv.set_params()` + `SubprocVecEnv` でポート番号を分けて複数インスタンス起動。  
ただし PIE では不可。パッケージビルドが前提（Phase 2-B 参照）。

---

### 2. gRPC（双方向ストリーミング）

```
Python (RLlib / SB3)
├── Worker 1 ↔ UE5 Instance 1
├── Worker 2 ↔ UE5 Instance 2
└── Worker 3 ↔ UE5 Instance 3
         ↓（ONNX export）
       UE5 C++ 推論（NNERuntimeORT）
```

**メリット**
- 高速な双方向通信（HTTP より低レイテンシ）
- 大量の観測データ転送に向く

**デメリット**
- UE5 側に gRPC プラグインの導入が必要（公式サポートなし）
- プロトコルの設計・実装コストが高い
- 複数インスタンスによる並列化はリソースを大量消費

---

### 3. UnrealCV

**メリット**
- Python API が整備済みで導入が比較的簡単
- 画像ベースの観測取得に強い（カメラ・深度・セグメンテーション）

**デメリット**
- UE5.4 対応が不確実（コミュニティ維持）
- 画像ベース前提の設計で数値ベース観測には過剰
- 更新頻度が低下傾向にある

---

### 4. Learning Agents（Epic 公式プラグイン）

```
UE5 エディタ（1プロセス）
└── ULearningAgentsManager
    ├── Agent 1
    ├── Agent 2
    └── ...（N個）
         ↓
   ULearningAgentsPolicy（NNI）
```

**メリット**
- Epic 公式プラグインで UE5.4 対応が確実
- 単一 UE5 インスタンス内で複数エージェントを並列訓練可能
- エディタ上で訓練の可視化・デバッグが可能

**デメリット**
- 学習アルゴリズムのカスタマイズに制約がある（独自アルゴリズムが使えない）
- 本プロジェクトが求める EUREKA 型 LLM 報酬ループとの統合が困難

---

## 一覧比較表

| 観点 | カスタムHTTP（採用） | gRPC | UnrealCV | Learning Agents |
|------|:-------------------:|:----:|:--------:|:---------------:|
| 導入コスト | 中 | 高 | 中 | 低 |
| UE5.4 対応 | ✅ | 自前で対応 | 不確実 | ✅ 確実（公式） |
| RLライブラリの自由度 | 高 | 高 | 高 | 中 |
| LLM報酬ループ対応 | ✅ 容易 | 容易 | 容易 | ❌ 困難 |
| 数値ベース観測 | 適 | 適 | 不向き | 適 |
| 並列訓練（PIE内） | ❌ | ❌ | ❌ | ✅ |
| 実機推論への移行 | ONNX + NNERuntimeORT | ONNX + NNERuntimeORT | 手動 | NNI 統合済み |
| 保守性 | 自己責任 | 自己責任 | コミュニティ | Epic 公式 |

---

## 本プロジェクトの採用理由

**カスタム HTTP サーバー** を採用。

理由：
1. EUREKA 型 LLM 報酬シェーピングループとの統合が容易（Python 側で報酬関数を差し替えるだけ）
2. SB3 / RLlib など任意の RL ライブラリを使用できる自由度
3. obs_schema API により観測次元を動的に変更できる（ゲームバランス実験に有利）
4. frame_skip を UE5 側の N-ループで実装し HTTP 往復コストを削減できる

---

## 訓練から推論への流れ（本プロジェクト）

```
[訓練]                              [実機推論]
Python (SB3 PPO) ←HTTP→ UE5 PIE
        ↓
  モデル保存 (SB3 .zip)
        ↓
  export_onnx.py で ONNX 変換
        ↓
  Content/Models/*.onnx を配置
        ↓
  UE5 NNERuntimeORT で推論（C++ のみで完結）
```
