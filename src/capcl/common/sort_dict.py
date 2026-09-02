# ruff: noqa: ANN401
from typing import Any, ClassVar


class SortDict(dict):
    MODEL_OUTPUT_ORDER: ClassVar[list[str]] = []

    def __init__(self, *kargs: Any, **kwargs: Any) -> None:
        super().__init__(*kargs, **kwargs)
        self.sort_dict()

    def sort_dict(self) -> None:
        if any(key not in self.keys() for key in self.MODEL_OUTPUT_ORDER):
            msg = f"Missing keys in the dict. Required keys are: {self.MODEL_OUTPUT_ORDER}\n"
            msg += f"Current keys are: {self.keys()}\n"
            msg += f"The difference is: {set(self.MODEL_OUTPUT_ORDER) - set(self.keys())}"

            raise ValueError(msg)
        if any(key not in self.MODEL_OUTPUT_ORDER for key in self.keys()):
            msg = f"Extra keys in the dict. Allowed keys are: {self.MODEL_OUTPUT_ORDER}"
            raise ValueError(msg)

        sorted_items = {key: self[key] for key in self.MODEL_OUTPUT_ORDER}
        self.clear()
        self.update(sorted_items)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self.keys():
            msg = (
                f"Key '{key}' not in MODEL_OUTPUT_ORDER. "
                f"Allowed keys are: {self.MODEL_OUTPUT_ORDER}"
            )
            raise ValueError(msg)
        super().__setitem__(key, value)
        self.sort_dict()
