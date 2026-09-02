import json
import logging
from pathlib import Path
from typing import Any

from accelerate.state import PartialState


class LocalLogger:
    """Collect params/metrics in memory and dump them as JSON under output_dir."""

    def __init__(
        self, experiment_name: str, run_name: str, output_dir: Path | None = None
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name

        self.logger = logging.getLogger(__name__)

        self.output_dir = output_dir

        self.metrics: dict[str, Any] = {}
        self.params: dict[str, Any] = {}

    def __del__(self) -> None:
        self.end_run()

    def start_run(self) -> None:
        return

    def end_run(self) -> None:
        if self.output_dir is None or (not self.metrics and not self.params):
            return
        output_path = Path(self.output_dir) / "run_log.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump({"params": self.params, "metrics": self.metrics}, f, indent=4)

    def delete_run(self) -> None:
        return

    def _is_main_process(self, ddp_state: PartialState | None) -> bool:
        if ddp_state is None:
            return True
        is_main: bool = ddp_state.is_main_process

        return is_main

    def log_metric(
        self,
        key: str,
        value: float,
        step: int,
        ddp_state: PartialState | None = None,
    ) -> None:
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append((step, value))
        _ = ddp_state

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "",
        ddp_state: PartialState | None = None,
    ) -> None:
        if prefix:
            metrics = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
        else:
            metrics = {k: float(v) for k, v in metrics.items()}

        for key, value in metrics.items():
            self.log_metric(key, value, step, ddp_state)

    def log_param(
        self, params: dict[str, Any], ddp_state: PartialState | None = None
    ) -> None:
        self.params.update(params)
        _ = ddp_state

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str,
        ddp_state: PartialState | None = None,
    ) -> None:
        _, _, _ = local_path, artifact_path, ddp_state

    def log_table(
        self,
        table: dict[str, Any],
        table_name: str = "table.json",
        ddp_state: PartialState | None = None,
    ) -> None:
        if not self._is_main_process(ddp_state):
            return
        if not table_name.endswith(".json"):
            table_name += ".json"

        output_path = (
            self.output_dir / table_name if self.output_dir else Path(table_name)
        )

        with output_path.open("w") as f:
            json.dump(table, f, indent=4)
