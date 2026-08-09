# Capture Core

本家 Vampire Survivors の画面を Windows DXGI 経由で取得し、後続の dataset writer へ型付き frame を渡す最小実装。

target window client area の識別から最新 frame の取得まで、保存・annotation・Artifact 昇格を含まない。これらは 04-02 dataset writer へ委譲する。

---

## アーキテクチャ

```
TargetProfile/config
       │  target_profile_hash, game_build_id
       ▼
 WindowLocator           ← Win32 API Protocol (本番/fake 差し替え可)
  ・process exe + exact window class/title でHWND解決
  ・PID、client rect、DPI、DXGI output mapping を記録
  ・1920×1080 以外・跨ぎ window はfail-closed
       │  ResolvedWindow
       ▼
 FrameCapture            ← CaptureBackend Protocol (DXcam/fake 差し替え可)
  ・DXcam: backend=dxgi, processor_backend=numpy, output_color=BGRA
  ・target 30 FPS、latest-only bounded queue
  ・focus loss / resize はfail-closed
       │  CapturedFrame
       ▼
04-02 DatasetWriter      ← 本 PR の対象外
```

## CapturedFrame 契約

```python
@dataclass(frozen=True)
class CapturedFrame:
    frame_bgra: np.ndarray            # uint8 [1080,1920,4]
    captured_monotonic_ns: int        # time.monotonic_ns()
    session_frame_index: int          # セッション開始から単調増加
    client_rect_screen_px: tuple[int, int, int, int]  # (left, top, right, bottom)
    foreground: bool                  # 取得時点でtarget windowが前面か
    target_profile_hash: str          # TargetProfile の canonical hash (hex64)
    game_build_id: str                # TargetProfile の build_id
```

フィールドの追加・変更は 04-02 consumer との compatibility break になる。変更時は 04-02 test を必ず再実行すること。

## DXcam 設定

| パラメータ | 値 |
|---|---|
| `backend` | `"dxgi"` |
| `processor_backend` | `"numpy"` |
| `output_color` | `"BGRA"` |
| target FPS | 30 |
| queue | latest-only bounded (size=1) |

## fail-closed 条件

以下はすべて例外を raise し、部分結果を返さない。

- target window が見つからない、または解像度が 1920×1080 以外
- window が複数モニターにまたがっている
- desktop 全体や別 window を capture しようとした場合
- focus loss（foreground HWND が target と一致しない）
- window resize（client rect が変わった）

## テスト

fake Win32 API (`FakeWin32Api`) と fake capture backend (`FakeCaptureBackend`) でPRが完結する。dxcam 実機・本家 VS プロセスは不要。

```bash
bash Tools/run-pytest.sh Tools/Deployment/tests -q -rs
```

capture 専用テストは `Tools/Deployment/tests/capture/` に配置している。

## 制約と後続フェーズへの委譲

| 項目 | 本 PR（04-01） | 後続（04-02 以降） |
|---|---|---|
| frame 取得 | ✅ | - |
| frame 永続化 | ❌ 含まない | ✅ 04-02 |
| dataset / annotation | ❌ 含まない | ✅ 04-02 |
| performance verdict（30FPS 実測） | ❌ 含まない | ✅ 04-02 |
| 実機 DXcam 動作確認 | ❌ 含まない（fake のみ） | ✅ 04-02 |
| multi-monitor 完全対応 | ❌ 含まない | ✅ 04-02 以降 |
