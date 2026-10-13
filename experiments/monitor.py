"""Read-only monitor. Never trains, restarts or relabels failures."""

import json
import time
from pathlib import Path


def main(run):
    """Read worker status and recent log activity without starting or restarting training."""
    run = Path(run)
    while True:
        status = (
            json.loads((run / "status.json").read_text())
            if (run / "status.json").exists()
            else {}
        )
        row = {"checked_at": time.time(), "worker": status, "read_only": True}
        print(json.dumps(row), flush=True)
        (run / "monitor_status.json").write_text(json.dumps(row, indent=2) + "\n")
        if (
            status.get("phase") in ("complete", "stopped")
            or (run / "stopped.json").exists()
        ):
            return
        time.sleep(30)
