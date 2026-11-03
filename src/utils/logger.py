"""
Simple experiment logger
"""

import json
from pathlib import Path
from typing import Dict, Any


class Logger:
    def __init__(self, log_dir: str = "results"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def log(self, metrics: Dict[str, Any]):
        self.history.append(metrics)
        # Optionally write to file every N steps
        if len(self.history) % 50 == 0:
            self.save()

    def save(self):
        with open(self.log_dir / "training_log.json", "w") as f:
            json.dump(self.history, f, indent=2)
