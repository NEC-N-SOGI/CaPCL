import contextlib
import warnings
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging

from capcl import registry
from capcl.common.runtime_env import set_environment_variables
from capcl.dataset import DatasetBuilder
from capcl.dataset.imgtxt_dataset import ImgTxtDataset
from capcl.metrics import CaptionCache
from capcl.models.model_manager import ModelManager
from capcl.tasks.task_cfg import CfgArgParser, TaskCfg
from capcl.trainer import HFTrainerBuilder

if TYPE_CHECKING:
    from capcl.models.model import ModelBuilder


# off all warnings
warnings.filterwarnings("ignore")
logger = logging.get_logger("transformers")

cfg_parser = CfgArgParser()
parser = cfg_parser.parser
parser.add_argument(
    "--task-root",
    type=str,
    default="/capcl/experiments/results/local/incident_flickr/qformer.bert/org_modx/lora/0.00100/constant/",
)

parser.add_argument("--target-dataset", type=str, default="")
parser.add_argument("--dataset-cfg", type=str, default="")
parser.add_argument("--do-captioning", type=bool, default=True)

args, _, _, _, _ = cfg_parser.parse_args()

task_root = Path(args.task_root) / "0_task/args.json"
cfg_parser.load_cache(task_root)

data_root = registry.get_datacfg_root(args.machine)
cfg_parser.task_config.dataset_path = str(
    Path(data_root) / Path(cfg_parser.task_config.dataset_path).name
)
cfg_parser.task_config.output_dir = args.task_root

optim_cfg, model_cfg, lora_cfg, task_cfg = (
    cfg_parser.optim_config,
    cfg_parser.model_config,
    cfg_parser.lora_config,
    cfg_parser.task_config,
)


if args.local_rank <= 0:
    logging.set_verbosity_info()
else:
    logging.set_verbosity_error()

# set default dtype = bfloat16
torch.set_default_dtype(torch.bfloat16)

result_prefix, result_root_dir = registry.get_result_path(args.machine)

set_environment_variables(result_path=Path(result_root_dir))


class Runner:
    @staticmethod
    def prepare(
        model_manager: ModelManager,
        task_id: int,
        org_task_config: TaskCfg,
        batch_size: int,
        datasets: dict[str, ImgTxtDataset],
    ) -> Path:
        logger.info("Preparing for task", extra={"task_id": task_id})
        task_config = deepcopy(org_task_config)
        task_config.output_dir = Runner.rename_output_dir(
            org_task_config.output_dir,
            task_id,
        )
        model_builder: ModelBuilder = model_manager.model_builder

        # perpare target for knowledge distillation

        caption_path = Path(task_config.output_dir) / "caption_cache"

        model_manager.model.set_n_gen_caps(task_config.num_gen_caps)
        cacher = CaptionCache(
            output_dir=caption_path,
            tokenizer=model_manager.tokenizer,
            add_dataset_name=True,
        )
        trainer, train_args = HFTrainerBuilder.build_for_prediction(
            model=model_manager.base_model,
            model_builder=model_builder,
            output_dir=task_config.output_dir,
            batch_size=batch_size,
            compute_metrics=cacher.compute_metrics,
            batch_eval_metrics=True,
            seed=(task_id + 1) * 100,
            pad_token_id=model_manager.pad_token_id,
        )

        # with profile(use_cuda=True, with_stack=True) as prof:
        # with torch.inference_mode():
        for dataset in datasets.values():
            _ = trainer.evaluate(dataset)

        # save dataset.json
        logger.info("Captioning Done\n Decoding...")

        model_manager.model.set_n_gen_caps(0)

        if train_args.distributed_state is None:
            return caption_path

        train_args.distributed_state.wait_for_everyone()
        return caption_path

    @staticmethod
    def rename_output_dir(
        output_dir: str,
        task_id: int,
    ) -> str:
        prefix = "eval_task"

        output_dir_path = Path(output_dir)
        new_output_dir_path = output_dir_path / f"{task_id}_{prefix}"
        new_output_dir_path.mkdir(parents=True, exist_ok=True)
        return str(new_output_dir_path)

    @staticmethod
    def run() -> None:
        model_manager = registry.get_model_manager(task_cfg.train_strategy)(
            model_cfg, lora_cfg
        )

        model_builder = model_manager.model_builder

        # dataset

        # trains are from original config
        train_dataset_keys = DatasetBuilder.get_training_dataset_keys(
            task_cfg.dataset_path
        )

        # evals are from the args if specified
        if args.dataset_cfg != "":
            task_cfg.dataset_path = args.dataset_cfg

        eval_datasets = DatasetBuilder.eval_build(
            datasets_path=task_cfg.dataset_path,
            img_processor=model_builder.eval_img_processor,
            machine=args.machine,
            tokenizer=model_manager.tokenizer,
        )
        if args.target_dataset != "":
            eval_datasets = {args.target_dataset: eval_datasets[args.target_dataset]}

        for task_id in range(-1, len(train_dataset_keys)):
            result_path = Path(task_cfg.output_dir) / f"{task_id}_task"

            if task_id > -1:
                # load the trained model
                last_model_dir: None | str = get_last_checkpoint(result_path)
                if last_model_dir is None:
                    msg = f"Checkpoint not found in {result_path}. Please check the path."
                    raise ValueError(msg)
                model_manager.from_pretrained(last_model_dir)

                context = contextlib.nullcontext()

            else:
                # if lora, deactivate the adapter. if ft, the manager return nullcontext
                context = model_manager.disable_adapter()

            with context:
                if args.do_captioning:
                    Runner.prepare(
                        model_manager,
                        task_id,
                        deepcopy(task_cfg),
                        256,
                        eval_datasets,
                    )

                copy_cfg = deepcopy(cfg_parser)
                copy_cfg.task_config.output_dir = Runner.rename_output_dir(
                    copy_cfg.task_config.output_dir, task_id
                )
                trainer, _ = HFTrainerBuilder.build(
                    train_dataset=next(iter(eval_datasets.values())),  # dummy
                    eval_datasets=eval_datasets,
                    model_builder=model_builder,
                    model=model_manager.model,
                    cfg=copy_cfg,
                    seed=task_id + 1,
                    pad_token_id=model_manager.pad_token_id,
                )

                logger.info("Evaluating for task", extra={"task_id": task_id})
                metrics = trainer.evaluate()
                logger.info(metrics)

            if task_id > -1:
                model_manager.update_base_model()

                state_dict = model_manager.base_state_dict()

                torch.save(
                    state_dict,
                    Path(copy_cfg.task_config.output_dir) / "model.pt",
                )


Runner.run()
