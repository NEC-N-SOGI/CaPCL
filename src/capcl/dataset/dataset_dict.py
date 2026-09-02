import json
from pathlib import Path
from typing import Self

from pydantic import (
    BaseModel,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from pydantic_core import from_json as pydantic_from_json

VALID_DATA_SPLIT = {"train", "val", "test"}


class DatasetModel(BaseModel):
    name: StrictStr
    images: list[StrictStr]
    texts: list[StrictStr]  # image texts to be learned
    txt2img: list[list[StrictInt]] = Field(
        ...,
        description="Label for image-to-text retrieval. len(label_i2t) == len(images)",
    )

    @model_validator(mode="after")
    def check_label_shape(self) -> Self:
        if len(self.txt2img) != len(self.texts):
            msg = f"DatasetModel Validation Error: Length of txt2img must be {len(self.texts)}. Got {len(self.txt2img)}"
            raise ValueError(msg)

        if any(max(label) >= len(self.images) for label in self.txt2img):
            msg = "DatasetModel Validation Error: Label index must be less than the number of images."
            raise ValueError(msg)

        return self

    @classmethod
    def from_json(cls, json_path: str | Path) -> Self:
        with Path(json_path).open("r") as f:
            return cls.model_validate(pydantic_from_json(f.read()))


class DatasetDictModel(BaseModel):
    dataset: dict[StrictStr, DatasetModel]  # manage dataset for each split
    target_metric: dict[StrictStr, StrictStr]

    @staticmethod
    def _datasplit_startwith(dataset_name: str) -> bool:
        return any(dataset_name.startswith(k) for k in VALID_DATA_SPLIT)

    @field_validator("dataset")
    @classmethod
    def check_dataset_keys(
        cls, dataset: dict[StrictStr, DatasetModel]
    ) -> dict[StrictStr, DatasetModel]:
        if not all(cls._datasplit_startwith(k) for k in dataset):
            msg = f"DatasetDictModel Validation Error: Dataset split must start with {VALID_DATA_SPLIT}"
            raise ValueError(msg)
        return dataset

    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
    ) -> Self:
        # load image-text dataset
        root = Path(json_path).parent
        with Path(json_path).open("r") as f:
            config = json.load(f)

        _dict = {}
        metrics = {}
        for data_id, data_json in config.items():
            # get target metric for each dataset
            target_metric = data_json.pop("target_metric", "mrr")

            has_test = False
            for split, data_path in data_json.items():
                # load train/val/test dataset
                dataset_model = DatasetModel.from_json(root / data_path)

                # register dataset model
                data_key = f"{split}_{data_id}_{dataset_model.name}"

                _dict[data_key] = dataset_model
                metrics[data_key] = target_metric

                if split == "test":
                    has_test = True

            if not has_test:
                msg = f"Each dataset must have a 'test' split. {data_id} does not."
                raise ValueError(msg)

        return cls(dataset=_dict, target_metric=metrics)
