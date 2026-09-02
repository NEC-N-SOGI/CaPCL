import json
import multiprocessing as mp
import warnings
from copy import deepcopy
from pathlib import Path

import torch
from transformers import TrainingArguments
from transformers.trainer_utils import TrainOutput, get_last_checkpoint
from transformers.utils import logging

from capcl import registry
from capcl.common.runtime_env import set_environment_variables
from capcl.dataset import DatasetBuilder, DatasetModel, ImgTxtDataset
from capcl.logger import LocalLogger
from capcl.metrics import CaptionCache, CLMetricsCalculator
from capcl.models.model_manager import ModelManager
from capcl.tasks.task_cfg import CfgArgParser, TaskCfg
from capcl.trainer import HFTrainerBuilder

mp.set_start_method("spawn", force=True)

# off all warnings
warnings.filterwarnings("ignore")
logger = logging.get_logger("transformers")
parser = CfgArgParser()  # argparse.ArgumentParser()

args, optim_cfg, model_cfg, lora_cfg, task_cfg = parser.parse_args()

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
    def rename_output_dir(
        output_dir: str,
        task_id: int,
        is_captioning: bool = False,
    ) -> str:
        prefix = "captioning" if is_captioning else "task"

        output_dir_path = Path(output_dir)
        new_output_dir_path = output_dir_path / f"{task_id}_{prefix}"
        new_output_dir_path.mkdir(parents=True, exist_ok=True)
        return str(new_output_dir_path)

    @staticmethod
    def is_dataset_updated(
        dataset_json_path: Path, dataset_model: DatasetModel
    ) -> bool:
        if not dataset_json_path.exists():
            return True

        current_dataset_json = dataset_model.model_dump()

        cached_dataset = DatasetModel.from_json(dataset_json_path)
        cached_dataset_json = cached_dataset.model_dump()

        # check keys
        current_keys = set(current_dataset_json.keys())
        cached_keys = set(cached_dataset_json.keys())
        if current_keys != cached_keys:
            return True

        # check name
        if dataset_model.name != cached_dataset.name:
            logger.info("name is different =============")
            return True

        # check images
        if set(dataset_model.images) != set(cached_dataset.images):
            logger.info("images are different =============")
            return True

        # check texts
        if set(dataset_model.texts) != set(cached_dataset.texts):
            logger.info("texts are different =============")
            return True

        # check txt2img
        if len(dataset_model.txt2img) != len(cached_dataset.txt2img):
            logger.info("txt2img length is different =============")
            return True

        for curr_t2i_list, cache_t2i_list in zip(
            dataset_model.txt2img, cached_dataset.txt2img, strict=True
        ):
            if set(curr_t2i_list) != set(cache_t2i_list):
                logger.info("txt2img is different =============")
                return True

        return False

    @staticmethod
    def is_cached(cache_root: Path, ids: list[int], prefix: str = "") -> bool:
        for id_ in ids:
            binpath = CaptionCache.imgid_to_binpath(id_, cache_root).absolute()

            if prefix != "":
                cache_path = binpath.parent / f"{prefix}_{binpath.name}"
            else:
                cache_path = binpath

            if not cache_path.exists():
                msg = (
                    f"Cache not found for id {id_} at {cache_path}. "
                    "Please run captioning first."
                )
                logger.warning(msg)
                return False
        return True

    @staticmethod
    def is_already_done(
        cache_root: Path,
        dataset_json_path: Path,
        dataset: ImgTxtDataset,
        remove_multi_texts: bool = False,
    ) -> bool:
        if Runner.is_dataset_updated(dataset_json_path, dataset.dataset):
            logger.info("dataset is updated")
            return False  # need to recompute, due to different dataset or no cache

        if not Runner.is_cached(cache_root, dataset.img_ids):
            logger.info("caption is not cached")
            return False

        if not Runner.is_cached(cache_root, dataset.img_ids, "img_feat"):
            logger.info("img_feat is not cached")
            return False

        if (not remove_multi_texts) and (
            not Runner.is_cached(cache_root, dataset.txt_ids, "text_feat")
        ):
            logger.info("text_feat is not cached")
            return False

        return True

    @staticmethod
    def prepare(
        model_manager: ModelManager,
        task_id: int,
        org_task_config: TaskCfg,
        dataset_key: str,
        batch_size: int,
        pad_token_id: int,
        remove_multi_texts: bool = False,
    ) -> Path:
        logger.info("Preparing for task", extra={"task_id": task_id})
        task_config = deepcopy(org_task_config)
        task_config.output_dir = Runner.rename_output_dir(
            org_task_config.output_dir,
            task_id,
            is_captioning=True,
        )
        model_builder = model_manager.model_builder

        # perpare target for knowledge distillation

        train_dataset = DatasetBuilder.build_from_split(
            dataset_path=task_cfg.dataset_path,
            caption_path=None,
            img_processor=model_builder.eval_img_processor,
            split=dataset_key,
            remove_multi_texts=remove_multi_texts,
            machine=args.machine,
            use_caption=False,
            use_prev_feat=False,
            tokenizer=model_manager.tokenizer,
        )

        caption_path = Path(task_config.output_dir) / "caption_cache"
        dataset_json_path = caption_path / "dataset.json"

        if (
            Runner.is_already_done(
                caption_path, dataset_json_path, train_dataset, remove_multi_texts
            )
            and org_task_config.resume_from_checkpoint
        ):
            logger.info("Already Done. Use cached caption")
            return caption_path

        model_manager.model.set_n_gen_caps(org_task_config.num_gen_caps)

        cacher = CaptionCache(output_dir=caption_path)
        trainer, train_args = HFTrainerBuilder.build_for_prediction(
            model=model_manager.base_model,
            model_builder=model_builder,
            output_dir=task_config.output_dir,
            batch_size=batch_size,
            compute_metrics=cacher.compute_metrics,
            batch_eval_metrics=True,
            seed=task_id * 100,
            pad_token_id=pad_token_id,
        )

        # with profile(use_cuda=True, with_stack=True) as prof:
        # with torch.inference_mode():
        with model_manager.disable_adapter():
            _ = trainer.evaluate(train_dataset)

        # save dataset.json
        is_main_process = (
            train_args.distributed_state is None
            or train_args.distributed_state.is_main_process
        )

        if is_main_process:
            dataset_json_path = caption_path / "dataset.json"
            with Path(dataset_json_path).open("w") as f:
                json.dump(train_dataset.dataset.model_dump(), f)

        logger.info("Captioning Done\n Decoding...")

        model_manager.model.set_n_gen_caps(0)

        Runner.wait_for_everyone(train_args)
        return caption_path

    @staticmethod
    def wait_for_everyone(
        train_args: TrainingArguments,
    ) -> None:
        if train_args.distributed_state is None:
            return
        train_args.distributed_state.wait_for_everyone()

    @staticmethod
    def save_args(train_args: TrainingArguments, parser: CfgArgParser) -> None:
        if train_args.distributed_state is None:
            parser.save_args()
            return

        if train_args.distributed_state.is_main_process:
            parser.save_args()

        Runner.wait_for_everyone(train_args)

    @staticmethod
    def run() -> None:
        model_manager = registry.get_model_manager(task_cfg.train_strategy)(
            model_cfg, lora_cfg
        )
        tokenizer = model_manager.tokenizer
        pad_token = model_manager.pad_token_id

        model_builder = model_manager.model_builder

        # dataset
        train_dataset_keys = DatasetBuilder.get_training_dataset_keys(
            task_cfg.dataset_path
        )
        logger.info("Training on", extra={"train_dataset_keys": train_dataset_keys})

        run_logger = LocalLogger(
            experiment_name=Path(task_cfg.dataset_path).stem,
            run_name=parser.run_name,
            output_dir=Path(task_cfg.output_dir),
        )

        # Continueal Learning Metrics
        cl_metrics_calculator = CLMetricsCalculator(
            dataset_path=Path(task_cfg.dataset_path),
        )

        # eval datasets
        eval_datasets = DatasetBuilder.eval_build(
            datasets_path=task_cfg.dataset_path,
            img_processor=model_builder.eval_img_processor,
            machine=args.machine,
            tokenizer=tokenizer,
        )

        accum_train_step = 0

        for task_id, dataset_key in enumerate(train_dataset_keys):
            caption_task_config = deepcopy(task_cfg)

            use_prev_feat = parser.task_config.gen_sim_distill != ""
            do_prepare = parser.task_config.use_caption or use_prev_feat

            caption_path = None
            if do_prepare:
                caption_path = Runner.prepare(
                    model_manager,
                    task_id,
                    caption_task_config,
                    dataset_key,
                    parser.train_args.batch_size,
                    remove_multi_texts=not use_prev_feat,
                    pad_token_id=pad_token,
                )

            # set outputdir
            copy_parser = deepcopy(parser)
            copy_parser.task_config.output_dir = Runner.rename_output_dir(
                copy_parser.task_config.output_dir, task_id
            )

            # resume setting
            is_resume = copy_parser.task_config.resume_from_checkpoint and (
                get_last_checkpoint(copy_parser.task_config.output_dir) is not None
            )
            is_cfg_updated, _ = copy_parser.is_args_updated()

            if is_resume and is_cfg_updated:
                copy_parser.load_cache()
                logger.info("[RESUME] ==== Load cached args ====")

            # train dataset
            train_dataset = DatasetBuilder.build_from_split(
                dataset_path=copy_parser.task_config.dataset_path,
                caption_path=caption_path,
                img_processor=model_builder.train_img_processor,
                split=dataset_key,
                tokenizer=tokenizer,
                machine=args.machine,
                use_caption=copy_parser.task_config.use_caption,
                use_prev_feat=use_prev_feat,
                captioning_model=copy_parser.task_config.captioning_model,
            )

            # trainer
            trainer, train_args = HFTrainerBuilder.build(
                train_dataset=train_dataset,
                eval_datasets=eval_datasets,
                model_builder=model_builder,
                model=model_manager.model,
                cfg=copy_parser,
                seed=task_id,
                pad_token_id=pad_token,
                accum_train_step=accum_train_step,
                run_logger=run_logger,
            )
            ddp_state = train_args.distributed_state

            logger.info("Training for task", extra={"task_id": task_id})
            model_manager.print_model_info()

            Runner.save_args(train_args, copy_parser)

            if task_id == 0:
                param_dict = parser.to_dict()
                param_dict.update(train_args.to_sanitized_dict())
                run_logger.log_param(param_dict, ddp_state=ddp_state)

                # pre evaluation
                with model_manager.disable_adapter():
                    metrics = trainer.evaluate()
                    run_logger.log_metrics(
                        metrics,
                        step=accum_train_step,
                        ddp_state=ddp_state,
                        prefix="eval_log",
                    )
                    cl_metrics_calculator.add_metrics(metrics)

            # Training
            Runner.wait_for_everyone(train_args)
            train_output: TrainOutput = trainer.train(
                resume_from_checkpoint=is_resume
            )
            accum_train_step += train_output.global_step + 1

            # post processing for the next task
            model_manager.update_base_model()

            # Evaluation
            Runner.wait_for_everyone(train_args)
            logger.info("Evaluating for task", extra={"task_id": task_id})
            with model_manager.disable_adapter():
                metrics = trainer.evaluate()
            cl_metrics = cl_metrics_calculator.calculate_metrics(metrics)
            metrics.update(cl_metrics)
            run_logger.log_metrics(
                metrics,
                step=accum_train_step,
                ddp_state=ddp_state,
                prefix="eval_log",
            )

            Runner.wait_for_everyone(train_args)

            if task_id == len(train_dataset_keys) - 1:
                metric_table = cl_metrics_calculator.make_table()
                run_logger.log_table(metric_table, ddp_state=ddp_state)

        run_logger.end_run()


if __name__ == "__main__":
    # run the runner
    Runner.run()
