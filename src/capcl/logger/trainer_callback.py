# ruff: noqa: ARG002, ANN401
from typing import Any

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from capcl.logger import LocalLogger


# callback for communication between nodes
class LogCallback(TrainerCallback):
    def __init__(self, run_logger: LocalLogger) -> None:
        super().__init__()
        self.run_logger = run_logger

    def _log_with_prefix(
        self,
        args: TrainingArguments,
        state: TrainerState,
        prefix: str,
        history_idx: int = -1,
        remove_first_directory: bool = False,
    ) -> None:
        if (
            args.distributed_state is not None
            and not args.distributed_state.is_main_process
        ):
            # If not the main process, skip logging
            return

        if len(state.log_history) == 0:
            return

        last_log_dict = state.log_history[history_idx]

        step = last_log_dict.get("step", None)
        if remove_first_directory:
            # Remove the first directory from the keys
            last_log_dict = {
                k.split("/", 1)[-1]: v for k, v in last_log_dict.items()
            }

        if step is None:
            msg = "No step found in the last log dictionary."
            raise ValueError(msg)
        step = int(step)

        self.run_logger.log_metrics(
            last_log_dict,
            step=step,
            prefix=prefix,
            ddp_state=args.distributed_state,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # global_step is incremented before calling on_step_end callbacks
        # (see transformers.trainer)
        # we use global_step - 1 to match the logging steps
        if (state.global_step - 1) % state.logging_steps != 0:
            return

        index = -1
        for i in range(-1, -len(state.log_history) - 1, -1):
            # if logging step and
            if not any("eval_test" in k for k in state.log_history[i]):
                index = i
                break

        self._log_with_prefix(
            args,
            state,
            prefix="train",
            remove_first_directory=True,
            history_idx=index,
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        # evaluation metrics
        self._log_with_prefix(args, state, prefix="validation", history_idx=-1)
