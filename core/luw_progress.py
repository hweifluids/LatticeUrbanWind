from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


class ProgressEmitter:
    def __init__(self, default_label: str):
        self.default_label = default_label
        self.enabled = os.environ.get("LUW_PROGRESS_MODE", "").strip().lower() == "gui"
        self._last_stage: str | None = None
        self._last_label: str | None = None
        self._last_detail: str | None = None
        self._last_bucket: int | None = None
        self._last_emit_at = 0.0

    def stage(self, stage: str, detail: str = "", *, label: str | None = None) -> None:
        self.emit(stage, detail=detail, label=label, indeterminate=True, force=True)

    def progress(
        self,
        stage: str,
        current: int,
        total: int,
        detail: str = "",
        *,
        label: str | None = None,
        force: bool = False,
    ) -> None:
        self.emit(
            stage,
            detail=detail,
            label=label,
            current=current,
            total=total,
            indeterminate=False,
            force=force or total <= 1 or current <= 1 or current >= total,
        )

    def complete(self, stage: str, detail: str = "", *, label: str | None = None) -> None:
        self.emit(stage, detail=detail, label=label, current=1, total=1, indeterminate=False, force=True)

    def emit(
        self,
        stage: str,
        *,
        detail: str = "",
        label: str | None = None,
        current: int | None = None,
        total: int | None = None,
        indeterminate: bool | None = None,
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return

        summary = label or self.default_label
        if indeterminate is None:
            indeterminate = current is None or total is None or total <= 0

        bucket = None
        if not indeterminate and current is not None and total is not None and total > 0:
            bucket = int((max(0, min(current, total)) * 200) / total)

        now = time.monotonic()
        if not force:
            same_header = (
                stage == self._last_stage
                and summary == self._last_label
                and detail == self._last_detail
                and bucket == self._last_bucket
            )
            if same_header and (now - self._last_emit_at) < 0.15:
                return

        payload: dict[str, Any] = {
            "stage": stage,
            "label": summary,
            "detail": detail,
            "indeterminate": bool(indeterminate),
        }
        if current is not None:
            payload["current"] = int(current)
        if total is not None:
            payload["total"] = int(total)

        stream = getattr(sys, "__stdout__", None) or sys.stdout
        stream.write("[[LUW_PROGRESS]]" + json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()

        self._last_stage = stage
        self._last_label = summary
        self._last_detail = detail
        self._last_bucket = bucket
        self._last_emit_at = now
