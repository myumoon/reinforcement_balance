"""focus-safe input lease controller の semantic-only launcher。
対象 PID/HWND を固定し、stdin の0〜8 actionだけを独立 Win32 helper へ送ります。
"""
from __future__ import annotations
import argparse
from pathlib import Path
from survivors.action_semantics import load_action_contract
from survivors.input.controller import HelperUnavailable, InputLeaseController
from survivors.target_profile import load_target_profile
def _arguments() -> argparse.Namespace:
    """target binding と audit path だけを command line から読む。
    任意 VK・text・shortcut・click 座標の option は意図的に提供しません。
    """
    parser = argparse.ArgumentParser(description="Survivors focus-safe semantic input lease helper")
    parser.add_argument("--target-pid", required=True, type=int)
    parser.add_argument("--target-hwnd", required=True, type=int)
    parser.add_argument("--audit", required=True, type=Path)
    return parser.parse_args()
def main() -> int:
    """default-disarmed helper を起動し stdin の semantic action を転送する。
    Ctrl+Shift+F12 で arm した後も focus/PID/HWND gate と75ms期限が各 action に適用されます。
    """
    args = _arguments()
    target = load_target_profile()
    action = load_action_contract()
    try:
        with InputLeaseController(
            target_hash=target.target_hash, action_hash=action.contract_hash,
            target_pid=args.target_pid, target_hwnd=args.target_hwnd, audit_path=args.audit,
        ) as controller:
            for line in iter(input, ""):
                controller.send_action(int(line.strip()))
    except (EOFError, KeyboardInterrupt):
        return 0
    except (HelperUnavailable, ValueError) as exc:
        print(f"input helper stopped: {exc}")
        return 2
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
