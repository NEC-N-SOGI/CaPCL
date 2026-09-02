import argparse
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
)
from transformers.utils import logging

from capcl import registry


class OptimConfig(BaseModel):
    # https://github.com/huggingface/transformers/blob/7f5077e53682ca855afc826162b204ebf809f1f9/src/transformers/training_args.py#L144
    optim: StrictStr = Field(default="sgd", description="Optimizer")

    learning_rate: StrictFloat = Field(default=5e-4, description="Learning rate")
    weight_decay: StrictFloat = Field(default=0.0, description="Weight decay")

    lr_scheduler_type: StrictStr = Field(
        default="cosine", description="Learning rate schedule type"
    )
    lr_scheduler_kwargs: dict = Field(
        default_factory=dict, description="Learning rate scheduler kwargs"
    )

    warmup_ratio: StrictFloat = Field(
        default=0.1, description="Warmup ratio for the learning rate scheduler"
    )

    max_grad_norm: StrictFloat = Field(
        default=100.0, description="Max gradient norm for gradient clipping"
    )
    gradient_accumulation_steps: StrictInt = Field(default=1, description="")

    num_train_epochs: StrictInt = Field(default=100, description="")

    model_config = ConfigDict(extra="forbid")  # permit extra fields

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--optim", type=str, default="")
        parser.add_argument("--learning_rate", type=float, default=-1)
        parser.add_argument("--weight_decay", type=float, default=-1)
        parser.add_argument("--lr_scheduler_type", type=str, default="")
        parser.add_argument("--lr_scheduler_kwargs", type=str, default="")
        parser.add_argument("--warmup_ratio", type=float, default=-1)
        parser.add_argument("--max_grad_norm", type=float, default=-1)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=-1)
        parser.add_argument("--num_train_epochs", type=int, default=-1)

    def update_from_command_line(self, args: argparse.Namespace) -> None:
        self.optim = args.optim if args.optim != "" else self.optim
        self.learning_rate = (
            args.learning_rate if args.learning_rate > -1 else self.learning_rate
        )
        self.weight_decay = (
            args.weight_decay if args.weight_decay > -1 else self.weight_decay
        )
        self.lr_scheduler_type = (
            args.lr_scheduler_type
            if args.lr_scheduler_type != ""
            else self.lr_scheduler_type
        )
        self.lr_scheduler_kwargs = (
            json.loads(args.lr_scheduler_kwargs)
            if args.lr_scheduler_kwargs != ""
            else self.lr_scheduler_kwargs
        )
        self.warmup_ratio = (
            args.warmup_ratio if args.warmup_ratio > -1 else self.warmup_ratio
        )
        self.max_grad_norm = (
            args.max_grad_norm if args.max_grad_norm > -1 else self.max_grad_norm
        )
        self.gradient_accumulation_steps = (
            args.gradient_accumulation_steps
            if args.gradient_accumulation_steps > -1
            else self.gradient_accumulation_steps
        )
        self.num_train_epochs = (
            args.num_train_epochs
            if args.num_train_epochs > -1
            else self.num_train_epochs
        )


class ModelConfig(BaseModel):
    model_class: StrictStr = Field(
        default="blip2_pretrain",
        description="Specify ModelBuilder class from capcl.registry.mapping.model_builders",
    )
    model_type: StrictStr = Field(
        default="coco",
        description="Name of the model. to be used for selecting model",
    )
    target_module_name: StrictStr = Field(
        description="Target modules for LORA", default="qformer.bert"
    )

    top_p: StrictFloat = Field(
        default=0.9,
        description="Top-p (nucleus) sampling parameter for caption generation",
    )
    top_k: StrictInt = Field(
        default=-1,
        description="Top-k sampling parameter for caption generation",
    )
    temperature: StrictFloat = Field(
        default=1.0,
        description="Temperature parameter for caption generation",
    )

    sdpa_backend: Literal["org", "torch"] = Field(
        default="org", description="SDPA backend for attention computation"
    )

    model_config = ConfigDict(extra="forbid")  # permit extra fields

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model_class", type=str, default="")
        parser.add_argument("--model_type", type=str, default="")
        parser.add_argument("--target_module_name", type=str, default="")
        parser.add_argument("--top_p", type=float, default=-1)
        parser.add_argument("--top_k", type=int, default=-1)
        parser.add_argument("--temperature", type=float, default=-1)
        parser.add_argument("--sdpa_backend", type=str, default="")

    def update_from_command_line(self, args: argparse.Namespace) -> None:
        self.model_class = (
            args.model_class if args.model_class != "" else self.model_class
        )
        self.model_type = (
            args.model_type if args.model_type != "" else self.model_type
        )
        self.target_module_name = (
            args.target_module_name
            if args.target_module_name != ""
            else self.target_module_name
        )

        self.top_p = args.top_p if args.top_p > -1 else self.top_p
        self.top_k = args.top_k if args.top_k > -1 else self.top_k
        self.temperature = (
            args.temperature if args.temperature > -1 else self.temperature
        )

        if self.top_p < 1.0 and self.top_k > 0:
            self.top_p = 1.0  # top_k takes precedence over top_p
            warn_msg = "Both top_p and top_k are set. Setting top_p to 1.0 to give precedence to top_k."
            logger = logging.get_logger("task_cfg")
            logger.warning(warn_msg)

        self.sdpa_backend = (
            args.sdpa_backend if args.sdpa_backend != "" else self.sdpa_backend
        )


class LoRAConfig(BaseModel):
    r: StrictInt = Field(default=16, description="Lora r")
    lora_alpha: StrictInt = Field(default=32, description="Lora alpha")
    lora_dropout: StrictFloat = Field(default=0.1, description="Lora dropout")
    merge_scale: StrictFloat = Field(
        default=0.5,
        description="Scale factor applied to LoRA weights only when merging into the base model",
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none", description="Lora bias"
    )

    model_config = ConfigDict(extra="forbid")  # permit extra fields

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--r", type=int, default=-1)
        parser.add_argument("--lora_alpha", type=int, default=-1)
        parser.add_argument("--lora_dropout", type=float, default=-1)
        parser.add_argument("--merge_scale", type=float, default=-1)
        parser.add_argument("--bias", type=str, default="")

    def update_from_command_line(self, args: argparse.Namespace) -> None:
        self.r = args.r if args.r > -1 else self.r
        self.lora_alpha = (
            args.lora_alpha if args.lora_alpha > -1 else self.lora_alpha
        )
        self.lora_dropout = (
            args.lora_dropout if args.lora_dropout > -1 else self.lora_dropout
        )
        self.merge_scale = (
            args.merge_scale if args.merge_scale > -1 else self.merge_scale
        )
        self.bias = args.bias if args.bias != "" else self.bias


class TrainArgs(BaseModel):
    batch_size: int = Field(
        default=64, description="Batch size for the image captioning task"
    )

    eval_steps: StrictFloat = Field(
        default=0.5, description="Evaluation step", ge=0.0, le=1.0
    )
    logging_steps: StrictInt = Field(default=10, description="Logging steps")

    bf16: StrictBool = Field(default=True, description="Use fp16")

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--batch_size", type=int, default=-1)
        parser.add_argument("--eval_steps", type=float, default=-1)
        parser.add_argument("--logging_steps", type=int, default=-1)
        parser.add_argument("--bf16", type=int, default=-1)

    def update_from_command_line(self, args: argparse.Namespace) -> None:
        self.batch_size = (
            args.batch_size if args.batch_size > -1 else self.batch_size
        )

        self.eval_steps = (
            args.eval_steps if args.eval_steps > -1 else self.eval_steps
        )

        self.logging_steps = (
            args.logging_steps if args.logging_steps > 0 else self.logging_steps
        )

        self.bf16 = bool(args.bf16) if args.bf16 != -1 else self.bf16


class TaskCfg(BaseModel):
    dataset_path: StrictStr = Field(
        default="",
        description="Path to the dataset json file. json file should be in the format of capcl.schema.dataset_dict.DatasetDictModel",
    )
    output_dir: StrictStr = Field(
        default="/path-to-results/",
        description="Directory to save the model and other outputs",
    )
    train_strategy: Literal["ft", "lora"] = Field(
        default="ft", description="Training strategy"
    )
    use_caption: StrictBool = Field(
        default=False, description="Whether to use caption for the task"
    )
    gen_sim_distill: Literal["", "none", "kl", "mse"] = Field(
        default="",
        description="Similarity distillation through the generated captions",
    )
    resume_from_checkpoint: StrictBool = Field(
        default=True, description="Whether to resume from the checkpoint"
    )

    w_cap_loss: StrictFloat = Field(
        default=0.0,
        description="Weight for the caption loss. 0.0 means no caption loss",
    )

    w_itc_gen_loss: StrictFloat = Field(
        default=0.0,
        description="Weight for the ITC generation loss. 0.0 means no ITC generation loss",
    )

    itc_gen_per_image: StrictBool = Field(
        default=True,
        description=(
            "Key the generated captions by image instead of by query text. A "
            "caption describes one image, so this is what it asserts; the legacy "
            "path reused the query's text_ids and label_t2i, which made "
            "Retriever.remove_duplicate keep a single caption row per batch and "
            "claim it matched every query positive. Set False to reproduce that."
        ),
    )

    w_itc_dense_loss: StrictFloat = Field(
        default=0.0,
        description="Weight for the ITC dense loss. 0.0 means no ITC dense loss",
    )

    w_dense_cap_loss: StrictFloat = Field(
        default=0.0,
        description="Weight for the dense caption loss. 0.0 means no dense caption loss",
    )
    w_itc_loss: StrictFloat = Field(
        default=1.0,
        description="Weight for the ITC loss. 0.0 means no ITC loss",
    )

    captioning_model: StrictStr = Field(
        default="",
        description="Captioning model to use for generating captions before training",
    )

    use_prev_txt_feat: StrictBool = Field(
        default=False,
        description="Whether to use text features by the previous model (=True) or the current model (=False).",
    )

    cap_reduction: Literal["mean", "min", "max"] = Field(
        default="mean", description="Reduction method for gen caption loss"
    )

    num_gen_caps: StrictInt = Field(
        default=5, description="Number of generated captions"
    )

    model_config = ConfigDict(extra="forbid")  # permit extra fields

    @classmethod
    def add_argument(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output_dir", type=str, default="")
        parser.add_argument("--train_strategy", type=str, default="")
        parser.add_argument("--use_caption", type=int, default=-1)
        parser.add_argument("--dataset_path", type=str, default="")
        parser.add_argument("--gen_sim_distill", type=str, default="")
        parser.add_argument("--resume_from_checkpoint", type=int, default=-1)
        parser.add_argument("--w_cap_loss", type=float, default=-1)
        parser.add_argument("--w_itc_gen_loss", type=float, default=-1)
        parser.add_argument(
            "--itc_gen_pairing",
            choices=["per-image", "legacy"],
            default="",
            help="生成キャプションを画像ごとに数えるか、旧挙動: クエリの id を流用",
        )
        parser.add_argument("--w_itc_dense_loss", type=float, default=-1)
        parser.add_argument("--w_itc_loss", type=float, default=-1)
        parser.add_argument("--use_prev_txt_feat", type=int, default=-1)
        parser.add_argument(
            "--captioning_model",
            type=str,
            default="",
            help="Captioning model to use",
        )
        parser.add_argument(
            "--w_dense_cap_loss",
            type=float,
            default=-1,
            help="Weight for the dense caption loss",
        )
        parser.add_argument(
            "--cap_reduction",
            type=str,
            default="",
            help="Reduction method for gen caption loss",
        )
        parser.add_argument(
            "--num_gen_caps",
            type=int,
            default=-1,
            help="Number of generated captions",
        )

    def update_from_command_line(self, args: argparse.Namespace) -> None:
        self.output_dir = (
            args.output_dir if args.output_dir != "" else self.output_dir
        )

        self.train_strategy = (
            args.train_strategy if args.train_strategy != "" else self.train_strategy
        )
        self.use_caption = (
            bool(args.use_caption) if args.use_caption != -1 else self.use_caption
        )
        self.dataset_path = (
            args.dataset_path if args.dataset_path != "" else self.dataset_path
        )

        if args.gen_sim_distill not in {"", "none"}:
            self.gen_sim_distill = args.gen_sim_distill

        self.resume_from_checkpoint = (
            bool(args.resume_from_checkpoint)
            if args.resume_from_checkpoint != -1
            else self.resume_from_checkpoint
        )

        self.w_cap_loss = (
            args.w_cap_loss if args.w_cap_loss > -1 else self.w_cap_loss
        )

        self.w_itc_gen_loss = (
            args.w_itc_gen_loss if args.w_itc_gen_loss > -1 else self.w_itc_gen_loss
        )

        if args.itc_gen_pairing != "":
            self.itc_gen_per_image = args.itc_gen_pairing == "per-image"

        self.w_itc_dense_loss = (
            args.w_itc_dense_loss
            if args.w_itc_dense_loss > -1
            else self.w_itc_dense_loss
        )

        self.w_itc_loss = (
            args.w_itc_loss if args.w_itc_loss > -1 else self.w_itc_loss
        )

        self.use_prev_txt_feat = (
            bool(args.use_prev_txt_feat)
            if args.use_prev_txt_feat != -1
            else self.use_prev_txt_feat
        )

        self.captioning_model = (
            args.captioning_model
            if args.captioning_model != ""
            else self.captioning_model
        )

        self.w_dense_cap_loss = (
            args.w_dense_cap_loss
            if args.w_dense_cap_loss > -1
            else self.w_dense_cap_loss
        )

        self.cap_reduction = (
            args.cap_reduction if args.cap_reduction != "" else self.cap_reduction
        )

        self.num_gen_caps = (
            args.num_gen_caps if args.num_gen_caps > -1 else self.num_gen_caps
        )


class CfgArgParser:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        # required arguments
        self.parser.add_argument(
            "--task_config",
            default="",
            type=str,
            help="Path to the task config file",
        )

        # optional arguments
        self.parser.add_argument(
            "--optim_config",
            type=str,
            default="",
            help="Path to the optimizer config file",
        )
        self.parser.add_argument(
            "--model_config",
            type=str,
            default="",
            help="Path to the model config file",
        )
        self.parser.add_argument(
            "--lora_config",
            type=str,
            default="",
            help="Path to the LoRA config file",
        )
        self.parser.add_argument("--trainargs_config", type=str, default="")

        self.parser.add_argument("--machine", type=str, default="local")

        self.parser.add_argument("--local_rank", type=int, default=-1)

        TaskCfg.add_argument(self.parser)
        OptimConfig.add_argument(self.parser)
        ModelConfig.add_argument(self.parser)
        LoRAConfig.add_argument(self.parser)
        TrainArgs.add_argument(self.parser)

        self.logger = logging.get_logger("transformers")

        self.run_name: str = ""  # to be set after parsing args

    def read_json(self, path: str) -> dict:
        if path == "":
            return {}

        if not Path(path).exists():
            data_path = registry.get_datacfg_root(self.machine)
            cfg_path = Path(data_path).parent / path
        else:
            cfg_path = Path(path)

        with Path(cfg_path).open("r") as f:
            cfg_dict = json.load(f)

        if isinstance(cfg_dict, str):
            cfg_dict = json.loads(cfg_dict)

        if not isinstance(cfg_dict, dict):
            msg = f"Invalid config file: {cfg_path}. Expected a json file."
            raise TypeError(msg)

        return cfg_dict

    def read_yaml(self, path: str) -> dict:
        if path == "":
            return {}

        if not Path(path).exists():
            data_path = registry.get_datacfg_root(self.machine)
            cfg_path = Path(data_path).parent / path
        else:
            cfg_path = Path(path)

        with Path(cfg_path).open("r") as f:
            cfg_dict = yaml.safe_load(f)

        if isinstance(cfg_dict, str):
            cfg_dict = yaml.safe_load(cfg_dict)

        if not isinstance(cfg_dict, dict):
            msg = f"Invalid config file: {cfg_path}. Expected a yaml file."
            raise TypeError(msg)

        return cfg_dict

    def read_from_file(self, path: str) -> dict:
        if path == "":
            return {}

        suffix = Path(path).suffix.lower()
        if suffix == ".json":
            return self.read_json(path)

        if suffix in {".yaml", ".yml"}:
            return self.read_yaml(path)

        msg = f"Unsupported config file format: {suffix}. Supported formats are .json, .yaml, .yml"
        raise ValueError(msg)

    def parse_args(  # noqa: C901, PLR0912
        self, args: list[str] | None = None
    ) -> tuple[
        argparse.Namespace,
        OptimConfig,
        ModelConfig,
        LoRAConfig,
        TaskCfg,
    ]:
        parsed_args = self.parser.parse_args(args)

        self.args = parsed_args
        self.local_rank = parsed_args.local_rank
        self.machine = parsed_args.machine

        # read from json files
        optim_args = self.read_from_file(parsed_args.optim_config)
        self.optim_config = OptimConfig(**optim_args)

        model_args = self.read_from_file(parsed_args.model_config)
        self.model_config = ModelConfig(**model_args)

        lora_args = self.read_from_file(parsed_args.lora_config)
        self.lora_config = LoRAConfig(**lora_args)

        train_args = self.read_from_file(parsed_args.trainargs_config)
        self.train_args = TrainArgs(**train_args)

        task_args = self.read_from_file(parsed_args.task_config)
        self.task_config = TaskCfg(**task_args)

        # update with command line arguments
        self.optim_config.update_from_command_line(parsed_args)
        self.model_config.update_from_command_line(parsed_args)
        self.lora_config.update_from_command_line(parsed_args)
        self.train_args.update_from_command_line(parsed_args)
        self.task_config.update_from_command_line(parsed_args)

        # to avoid the evaluation at the end of the training
        self.train_args.eval_steps += 0.01

        # set dataset path with machine specific path
        root = Path(registry.get_datacfg_root(parsed_args.machine))
        self.task_config.dataset_path = str(
            root / Path(self.task_config.dataset_path).name
        )

        self.set_outputdir()

        self.raise_inconsistent_params()

        optim_name = f"{self.optim_config.optim}_{self.optim_config.learning_rate:.5f}_{self.optim_config.lr_scheduler_type}_wu{self.optim_config.warmup_ratio:.2f}"
        model_name = (
            f"{self.model_config.model_class}_{self.model_config.model_type}"
        )
        target_module = self.model_config.target_module_name.replace(".", "_")
        task_name = f"{self.task_config.train_strategy}"

        if self.task_config.use_caption:
            task_name += "_cap"
            if self.task_config.cap_reduction != "mean":
                task_name += f"_{self.task_config.cap_reduction}"

            if self.task_config.num_gen_caps != 5:  # noqa: PLR2004
                task_name += f"_ncap{self.task_config.num_gen_caps}"

            if self.model_config.top_k > 0:
                task_name += f"_topk{self.model_config.top_k}"

            if self.model_config.top_p < 1.0:
                task_name += f"_topp{self.model_config.top_p}"

            if self.model_config.temperature != 1.0:
                task_name += f"_temp{self.model_config.temperature:.2f}"
        else:
            task_name += "_org"

        if self.task_config.w_itc_loss < 1e-6:  # noqa: PLR2004
            task_name += "_noitc"

        task_name += "_none"

        if self.task_config.gen_sim_distill != "":
            task_name += f"_gen_{self.task_config.gen_sim_distill}"

        if self.task_config.w_itc_gen_loss > 0:
            if self.task_config.use_prev_txt_feat:
                task_name += "_itcgen_prev"
            else:
                task_name += "_itcgen"

        if self.task_config.w_itc_dense_loss > 0:
            task_name += "_itcdense"

        if (
            self.task_config.w_dense_cap_loss > 0
            and self.task_config.captioning_model != ""
        ):
            task_name += f"_densecap_{self.task_config.captioning_model}_{self.task_config.w_dense_cap_loss:.2f}"
            if self.task_config.cap_reduction != "mean":
                task_name += f"_{self.task_config.cap_reduction}"

        self.run_name = (
            f"{self.machine}_{task_name}_{model_name}_{target_module}_{optim_name}"
        )

        return (
            parsed_args,
            self.optim_config,
            self.model_config,
            self.lora_config,
            self.task_config,
        )

    def raise_inconsistent_params(self) -> None:
        # TODO(dev): remove use_caption and automatically set it
        if self.task_config.w_cap_loss > 0 and not self.task_config.use_caption:
            msg = "Caption loss is set but use_caption is False. Set use_caption to True to use caption loss."
            raise ValueError(msg)

        if self.task_config.w_itc_gen_loss > 0 and not self.task_config.use_caption:
            msg = "ITC generation loss is set but use_caption is False. Set use_caption to True to use ITC generation loss."
            raise ValueError(msg)

        if (
            self.task_config.gen_sim_distill != ""
            and not self.task_config.use_caption
        ):
            msg = "Similarity distillation is set but use_caption is False. Set use_caption to True to use similarity distillation."
            raise ValueError(msg)

    def set_outputdir(self) -> None:  # noqa: C901,PLR0912
        # set output dir with machine specific path
        dataset_stem = Path(self.task_config.dataset_path).stem
        target_module_name = self.model_config.target_module_name

        method = "cap" if self.task_config.use_caption else "org"

        if method == "cap":
            if self.task_config.cap_reduction != "mean":
                method += f"_{self.task_config.cap_reduction}"

            if self.task_config.num_gen_caps != 5:  # noqa: PLR2004
                method += f"_ncap{self.task_config.num_gen_caps}"

            if self.model_config.top_k > 0:
                method += f"_topk{self.model_config.top_k}"
            if self.model_config.top_p < 1.0:
                method += f"_topp{self.model_config.top_p}"
            if self.model_config.temperature != 1.0:
                method += f"_temp{self.model_config.temperature:.2f}"

        if self.task_config.w_itc_gen_loss > 0:
            if self.task_config.use_prev_txt_feat:
                method = f"{method}_itcgen_prev"
            else:
                method = f"{method}_itcgen"

        if self.task_config.w_itc_dense_loss > 0:
            method = f"{method}_itcdense"

        if self.task_config.gen_sim_distill != "":
            method = f"{method}_gen_{self.task_config.gen_sim_distill}"

        if "cap" in method:
            method = f"{method}_{self.task_config.w_cap_loss:.2f}"

        if self.task_config.w_itc_loss < 1e-6:  # noqa: PLR2004
            method += "_noitc"

        result_prefix, result_root_dir = registry.get_result_path(self.machine)

        lr = f"{self.optim_config.learning_rate:.5f}"
        if self.optim_config.optim != "sgd":
            lr = f"{self.optim_config.optim}_{lr}"

        if self.optim_config.weight_decay > 0:
            lr = f"{lr}_wd{self.optim_config.weight_decay:.5f}"

        if (
            self.task_config.w_dense_cap_loss > 0
            and self.task_config.captioning_model != ""
        ):
            method = f"{method}_densecap_{self.task_config.captioning_model}_{self.task_config.w_dense_cap_loss:.2f}"
            if self.task_config.cap_reduction != "mean":
                method += f"_{self.task_config.cap_reduction}"

        lr_schedulrer = self.optim_config.lr_scheduler_type

        if (
            self.model_config.model_class == "blip2_pretrain"
            and self.model_config.model_type == "coco"
        ):
            self.task_config.output_dir = self.task_config.output_dir.replace(
                result_prefix,
                str(
                    Path(result_root_dir)
                    / dataset_stem
                    / target_module_name
                    / method
                    / self.task_config.train_strategy
                    / lr
                    / lr_schedulrer
                ),
            )
            return
        self.task_config.output_dir = self.task_config.output_dir.replace(
            result_prefix,
            str(
                Path(result_root_dir)
                / dataset_stem
                / f"{self.model_config.model_class}_{self.model_config.model_type}"
                / target_module_name
                / method
                / self.task_config.train_strategy
                / lr
                / lr_schedulrer
            ),
        )

    def to_dict(self) -> dict:
        all_dict = self.optim_config.model_dump()
        all_dict.update(self.model_config.model_dump())
        all_dict.update(self.lora_config.model_dump())
        all_dict.update(self.task_config.model_dump())

        return all_dict

    def to_train_args(self) -> dict:
        train_args = self.optim_config.model_dump()

        train_args["per_device_train_batch_size"] = self.train_args.batch_size
        train_args["per_device_eval_batch_size"] = self.train_args.batch_size

        train_args.update(self.train_args.model_dump())

        del train_args["batch_size"]

        return train_args

    @property
    def cache_path(self) -> Path:
        return Path(self.task_config.output_dir) / "args.json"

    def save_args(self) -> None:
        is_updated, _ = self.is_args_updated()
        if is_updated:
            self.logger.info(
                "Overwriting the existing args.json file:",
                extra={"cache_path": str(self.cache_path)},
            )

        arg_dict = self.to_dict()

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(arg_dict, f, indent=4)

    def compare_cached_args(self) -> dict:
        new_args = self.to_dict()
        if not self.cache_path.exists():
            return new_args

        cached_args = self.read_json(str(self.cache_path))

        return {
            key: value
            for key, value in new_args.items()
            if key not in cached_args or value != cached_args[key]
        }

    def is_args_updated(self) -> tuple[bool, dict]:
        diff = self.compare_cached_args()
        return len(diff) > 0, diff

    def load_cache(self, cache_path: Path | None = None) -> None:
        if (not self.cache_path.exists()) and (cache_path is None):
            return

        _cache_path = self.cache_path
        if cache_path is not None:
            _cache_path = cache_path

        with _cache_path.open("r", encoding="utf-8") as f:
            cached_args = json.load(f)

        optim_args = {}
        model_args = {}
        lora_args = {}
        task_args = {}

        for key, value in cached_args.items():
            if key in OptimConfig.model_fields:
                optim_args[key] = value
            elif key in ModelConfig.model_fields:
                model_args[key] = value
            elif key in LoRAConfig.model_fields:
                lora_args[key] = value
            elif key in TaskCfg.model_fields:
                task_args[key] = value

        self.optim_config = OptimConfig(**optim_args)
        self.model_config = ModelConfig(**model_args)
        self.lora_config = LoRAConfig(**lora_args)
        self.task_config = TaskCfg(**task_args)

        self.set_outputdir()
