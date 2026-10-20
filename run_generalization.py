"""Run the reproducible domain-held-out experiment and save its results."""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.generalization import run_domain_held_out


if __name__ == "__main__":
    result = run_domain_held_out()
    Path("reports").mkdir(exist_ok=True)
    result["experiment_timestamp"] = datetime.now(timezone.utc).isoformat()
    Path("reports/generalization_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
