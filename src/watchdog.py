#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kill a training PID on sustained instability.")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("--alpha-max", type=float, default=0.08)
    parser.add_argument("--alpha-growth", type=float, default=2.0)
    parser.add_argument("--grad-norm", type=float, default=10.0)
    parser.add_argument("--grad-patience", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--state-file", type=Path, default=None)
    return parser.parse_args()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate(pid: int, reason: Dict[str, Any], state_file: Optional[Path]) -> None:
    payload = {"event": "watchdog_kill", "pid": pid, "reason": reason, "time": time.time()}
    text = json.dumps(payload, sort_keys=True)
    print(text, flush=True)
    if state_file is not None:
        state_file.write_text(text + "\n", encoding="utf-8")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(30)
    if process_alive(pid):
        os.kill(pid, signal.SIGKILL)


def read_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def train_events_since(log_path: Path, offset: int) -> tuple[int, list[Dict[str, Any]]]:
    if not log_path.exists():
        return offset, []
    events = []
    with log_path.open("rb") as handle:
        handle.seek(offset)
        for raw in handle:
            try:
                item = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if item.get("event") == "train":
                events.append(item)
        return handle.tell(), events


def main() -> None:
    args = parse_args()
    offset = 0
    first_alpha_max: Optional[float] = None
    high_grad_count = 0
    while True:
        pid = read_pid(args.pid_file)
        if pid is not None and not process_alive(pid):
            print(json.dumps({"event": "watchdog_exit", "reason": "pid_exited", "pid": pid}), flush=True)
            return

        offset, events = train_events_since(args.log, offset)
        for event in events:
            step = int(event.get("step", -1))
            grad_norm = float(event.get("grad_norm") or 0.0)
            alpha_stats = event.get("alpha_stats") or {}
            alpha_max = float(alpha_stats.get("max") or 0.0)
            if first_alpha_max is None and alpha_max > 0.0:
                first_alpha_max = alpha_max

            if grad_norm > args.grad_norm:
                high_grad_count += 1
            else:
                high_grad_count = 0

            reason: Optional[Dict[str, Any]] = None
            if alpha_max > args.alpha_max:
                reason = {"trigger": "alpha_max", "step": step, "alpha_max": alpha_max}
            elif first_alpha_max is not None and alpha_max > first_alpha_max * args.alpha_growth:
                reason = {
                    "trigger": "alpha_growth",
                    "step": step,
                    "alpha_max": alpha_max,
                    "first_alpha_max": first_alpha_max,
                }
            elif high_grad_count >= args.grad_patience:
                reason = {
                    "trigger": "grad_norm",
                    "step": step,
                    "grad_norm": grad_norm,
                    "count": high_grad_count,
                }

            if reason is not None:
                if pid is None:
                    reason["pid_missing"] = True
                    print(json.dumps({"event": "watchdog_blocked", "reason": reason}), flush=True)
                    return
                terminate(pid, reason, args.state_file)
                return

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
