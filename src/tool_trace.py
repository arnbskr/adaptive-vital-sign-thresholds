"""Auditable tool-call tracing for the Phase 2 agent.

Every deterministic tool call the agent makes is recorded as one trace entry
(tool name, inputs, outputs, status, timestamp, latency). The full trace is
attached to the final answer and persisted to ``data/agent_traces/`` so each
response is reproducible and auditable.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import AGENT_TRACES_DIR, ensure_data_directories


def _summarize(value: Any, max_length: int = 600) -> str:
    """Short, JSON-ish preview of an input/output payload for table display."""

    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > max_length:
        return text[: max_length - 1] + "…"
    return text


class ToolTrace:
    """Collects an ordered list of tool-call records."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def record(self, tool_name: str, inputs: dict[str, Any], func: Callable[[], Any]) -> Any:
        """Run ``func`` (a zero-arg callable), timing it and capturing the result.

        On success the output is stored and returned. On error the entry is
        marked ``status="error"`` and the exception is re-raised so the agent
        can decide how to degrade.
        """

        step = len(self.entries) + 1
        started = time.perf_counter()
        timestamp = datetime.now().isoformat(timespec="seconds")
        try:
            output = func()
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self.entries.append(
                {
                    "step": step,
                    "tool_name": tool_name,
                    "inputs": inputs,
                    "outputs": output,
                    "outputs_summary": _summarize(output),
                    "status": "success",
                    "timestamp": timestamp,
                    "latency_ms": latency_ms,
                }
            )
            return output
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            self.entries.append(
                {
                    "step": step,
                    "tool_name": tool_name,
                    "inputs": inputs,
                    "outputs": {"error": str(exc)},
                    "outputs_summary": f"error: {exc}",
                    "status": "error",
                    "timestamp": timestamp,
                    "latency_ms": latency_ms,
                }
            )
            raise

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.entries)

    def success_rate(self) -> float:
        if not self.entries:
            return 0.0
        ok = sum(1 for entry in self.entries if entry["status"] == "success")
        return round(ok / len(self.entries), 4)

    def average_latency_ms(self) -> float:
        if not self.entries:
            return 0.0
        return round(sum(entry["latency_ms"] for entry in self.entries) / len(self.entries), 2)

    def tool_names(self) -> list[str]:
        return [entry["tool_name"] for entry in self.entries]


def save_trace(payload: dict[str, Any], timestamp: datetime | None = None) -> Path:
    """Persist a full agent response (answer + trace) to data/agent_traces/."""

    ensure_data_directories()
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    path = Path(AGENT_TRACES_DIR) / f"agent_trace_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return path
