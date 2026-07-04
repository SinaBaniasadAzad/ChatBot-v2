"""ثبتِ ماندگارِ تیکت‌های نهایی + تولیدِ شمارهٔ پیگیری.

JSONL append-only (مثلِ interaction log) — بعداً با ITSM واقعی جایگزین می‌شود بدون
تغییرِ رابط. شمارهٔ پیگیری: TKT-<سال>-<شمارندهٔ ۵رقمی> که با اسکنِ فایل در شروع
بازیابی می‌شود؛ thread-safe برای چند workerِ همزمانِ یک پروسه.

توجه: خروجی حاوی PII است و باید داخلِ logs/ (gitignore) بماند.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class TicketStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._year = datetime.now(timezone.utc).year
        self._seq = self._scan_last_seq()

    def _scan_last_seq(self) -> int:
        last = 0
        prefix = f"TKT-{self._year}-"
        try:
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        ref = json.loads(line).get("reference", "")
                    except json.JSONDecodeError:
                        continue
                    if ref.startswith(prefix):
                        try:
                            last = max(last, int(ref[len(prefix):]))
                        except ValueError:
                            continue
        except OSError:
            pass
        return last

    def submit(
        self,
        *,
        employee_id: str,
        first_name: str,
        last_name: str,
        summary: str,
        description: str,
        labels: dict,
        needs_review: bool = False,
        session_id: str | None = None,
    ) -> dict:
        with self._lock:
            self._seq += 1
            record = {
                "reference": f"TKT-{self._year}-{self._seq:05d}",
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "employee_id": employee_id,
                "first_name": first_name,
                "last_name": last_name,
                "summary": summary,
                "description": description,
                "labels": labels,
                "needs_review": needs_review,
                "session_id": session_id,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
