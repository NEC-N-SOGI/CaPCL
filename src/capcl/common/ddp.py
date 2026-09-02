# ruff: noqa: ARG002, ANN401
from typing import Any

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


# callback for communication between nodes
class SyncOnStepEndCallback(TrainerCallback):
    def _wait_for_everyone(self, args: TrainingArguments) -> None:
        if args.distributed_state is None:
            return
        args.distributed_state.wait_for_everyone()

    def _empty_cache(self, args: TrainingArguments) -> None:
        torch.cuda.empty_cache()

        self._wait_for_everyone(args)

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._empty_cache(args)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._empty_cache(args)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._empty_cache(args)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._wait_for_everyone(args)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._empty_cache(args)

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs: Any,
    ) -> None:
        self._empty_cache(args)
