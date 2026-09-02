from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import torch
import transformers
from peft import PeftModel
from torch.nn.parallel import DistributedDataParallel
from transformers import Trainer
from transformers.trainer_pt_utils import AcceleratorConfig
from transformers.trainer_utils import speed_metrics

from capcl import registry
from capcl.common.ddp import SyncOnStepEndCallback
from capcl.common.sort_dict import SortDict
from capcl.dataset import BatchImgTxtOutput, ImgTxtCollator
from capcl.dataset.imgtxt_dataset import ImgTxtDataset
from capcl.logger import LogCallback
from capcl.logger.local import LocalLogger
from capcl.loss import (
    DistillInput,
    InfoNCECrossModalLoss,
    NextTokenPredictionLoss,
    SigmoidLoss,
)
from capcl.metrics import MetricCalculator
from capcl.models import (
    BaseMixin,
    BaseModel,
    BasePeftLoraModel,
    ModelOutput,
)
from capcl.models.model import ModelBuilder
from capcl.models.retriever import Retriever
from capcl.tasks.task_cfg import CfgArgParser

LABEL_VARS = ["label_t2i", "text_ids", "image_ids"]


class TrainerOutput(SortDict):
    txt_feat: torch.Tensor
    img_feat: torch.Tensor
    caption_tokens: torch.Tensor
    dataset_names: torch.Tensor
    gen_txt_feat: torch.Tensor
    loss: torch.Tensor
    loss_itc: torch.Tensor
    loss_itc_gen: torch.Tensor
    loss_new_lm: torch.Tensor
    loss_gen_lm: torch.Tensor
    loss_gen_sim_distill: torch.Tensor
    loss_dense_cap: torch.Tensor
    loss_itc_dense: torch.Tensor

    MODEL_OUTPUT_ORDER: ClassVar[list[str]] = [
        "loss",
        "txt_feat",
        "img_feat",
        "caption_tokens",
        "dataset_names",
        "gen_txt_feat",
        "loss_itc",
        "loss_itc_gen",
        "loss_new_lm",
        "loss_gen_lm",
        "loss_gen_sim_distill",
        "loss_dense_cap",
        "loss_itc_dense",
    ]

    def __init__(self, *kargs: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(
            *kargs,
            **kwargs,  # additional outputs by the derivative models
        )
        self.__check_order()

    def __check_order(self) -> None:
        if self.MODEL_OUTPUT_ORDER[0] != "loss":
            msg = f"First key of MODEL_OUTPUT_ORDER must be 'loss'. Got {self.MODEL_OUTPUT_ORDER[0]}"
            raise ValueError(msg)

        # The following order is required by
        # - metrics.MetricCalculator
        # - metrics.CaptionCache
        if self.MODEL_OUTPUT_ORDER[1] != "txt_feat":
            msg = f"Second key of MODEL_OUTPUT_ORDER must be 'txt_feat'. Got {self.MODEL_OUTPUT_ORDER[1]}"
            raise ValueError(msg)

        if self.MODEL_OUTPUT_ORDER[2] != "img_feat":
            msg = f"Second key of MODEL_OUTPUT_ORDER must be 'img_feat'. Got {self.MODEL_OUTPUT_ORDER[2]}"
            raise ValueError(msg)

        if self.MODEL_OUTPUT_ORDER[3] != "caption_tokens":
            msg = f"Third key of MODEL_OUTPUT_ORDER must be 'caption_tokens'. Got {self.MODEL_OUTPUT_ORDER[3]}"
            raise ValueError(msg)

        if self.MODEL_OUTPUT_ORDER[4] != "dataset_names":
            msg = f"Fourth key of MODEL_OUTPUT_ORDER must be 'dataset_names'. Got {self.MODEL_OUTPUT_ORDER[4]}"
            raise ValueError(msg)


class HFTrainer(Trainer):  # ty: ignore[unsupported-base]
    def __init__(
        self,
        img_size: tuple[int, int],
        mixin: BaseMixin,
        gen_sim_distill: str,
        w_cap_loss: float,
        w_itc_gen_loss: float,
        w_itc_dense_loss: float,
        w_dense_cap_loss: float,
        w_itc_loss: float,
        use_prev_txt_feat: bool,
        itc_gen_per_image: bool = True,
        temperature: float | tuple[float, float] = 1.0,
        accum_train_step: int = 0,
        cap_reduction: str = "mean",
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        # a mixin class that has prepare_input method to organize inputs for each model.
        self.mixin = mixin
        self.w_cap_loss = torch.tensor(w_cap_loss)
        self.w_itc_gen_loss = torch.tensor(w_itc_gen_loss)
        self.w_itc_dense_loss = torch.tensor(w_itc_dense_loss)
        self.accum_train_step = accum_train_step
        self.w_itc_loss = torch.tensor(w_itc_loss)
        self.use_prev_txt_feat = use_prev_txt_feat
        self.itc_gen_per_image = itc_gen_per_image
        self.w_dense_cap_loss = torch.tensor(w_dense_cap_loss)

        self.lm_loss_fn = NextTokenPredictionLoss()
        self.lm_loss_gen_fn = NextTokenPredictionLoss(cap_reduction=cap_reduction)

        if isinstance(temperature, torch.nn.parameter.Parameter):
            temperature = temperature.detach().cpu().item()  # type: ignore[unreachable]

        if isinstance(temperature, float):
            self.info_nce_fn = InfoNCECrossModalLoss(temperature)
        elif isinstance(temperature, tuple):
            self.info_nce_fn = SigmoidLoss(temperature)
        else:
            msg = f"temperature must be float or tuple of float. Got {type(temperature)}"  # type: ignore[unreachable]
            msg += f" with value {temperature}"
            raise TypeError(msg)

        if gen_sim_distill == "":
            self.loss_gen_sim_distill_fn = None
        else:
            use_infonce_loss = isinstance(self.info_nce_fn, InfoNCECrossModalLoss)
            self.loss_gen_sim_distill_fn = registry.get_sim_distill_loss(
                gen_sim_distill
            )(False, apply_softmax=use_infonce_loss)  # noqa: FBT003 # dont use img-intra modal loss

    def _prepare_inputs(
        self, inputs: dict[str, torch.Tensor | Any]
    ) -> dict[str, BatchImgTxtOutput]:
        # inputs: output of dataset.ImgTxtCollator, i.e., BatchImgTxtOutput(TypedDict)

        # _prepare_inputs cast each torch.Tensor to the correct dtype and device.
        processed_inputs: BatchImgTxtOutput = super()._prepare_inputs(inputs)  # pyright: ignore[reportAssignmentType]

        # prepare_input apply the model specific preprocessing to the inputs.
        model_input: BatchImgTxtOutput = self.mixin.prepare_input(processed_inputs)

        # the return must be dict, because the return value will be passed to model.forward(**inputs) in compute_loss, evaluation, and prediction.
        # wrap the model_input with the key "batch" to match the model.forward(**inputs) signature.
        # i.e., all the models' signature is forward(self, batch: BatchImgTxtOutput, **kwargs: Any) -> ModelOutput
        model_input_dict: dict[str, Any | BatchImgTxtOutput] = {"batch": model_input}

        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        if not has_labels:
            return model_input_dict

        # add labels to the dict, as transformers expects labels to be in the input dict.
        # see transformers.trainer.py line 4432.
        for k in self.label_names:
            model_input_dict[k] = inputs[k]

        return model_input_dict

    @staticmethod
    def pair_generated_captions(
        image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        label_t2i: torch.Tensor,
        *,
        per_image: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Ids and labels for captions that each describe ONE image.

        The query ITC is handed `text_ids` -- a single id shared by every query
        positive in the batch -- and `label_t2i`, which marks every positive
        image as a match for that one query. Reusing both for the generated
        captions is wrong twice over:

        - `Retriever.remove_duplicate` keeps one row per distinct text id, so
          7 of 8 per-image captions are dropped and `sim_t2i` collapses back to
          `[1, bs]`. `soft_nn_loss` then sees an i2t matrix with one column per
          row, whose positive/total ratio is 1, so that half of the loss is
          exactly 0 -- the same dead term the single-text query ITC has.
        - the label claims a caption matches all 8 query positives, when it
          describes only the image it was generated from.

        Keying by image id gives one row per distinct image (repeated images in a
        multi-text task still collapse to one row, which is correct -- the
        caption is the same) and an identity label. `per_image=False` restores
        the old behaviour for comparison.
        """
        if not per_image:
            return text_ids, label_t2i

        keep = text_ids >= 0
        gen_txt_ids = torch.where(keep, image_ids, torch.full_like(image_ids, -1))
        # label_t2i の列はデータセット全体の画像インデックス。自分の画像だけを正例にする。
        gen_label = torch.zeros_like(label_t2i)
        rows = torch.arange(len(image_ids), device=label_t2i.device)[keep]
        gen_label[rows, image_ids[keep].to(label_t2i.device)] = 1
        return gen_txt_ids, gen_label

    def compute_itc_losses(
        self,
        outputs: ModelOutput,
        batch: BatchImgTxtOutput,
        model: torch.nn.Module,
        inputs: dict,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        dtype = outputs.img_feat.dtype

        # use only the [CLS] token for ITC loss when return_all_word_tokens is True
        if isinstance(model, DistributedDataParallel):
            return_all_word_tokens = model.module.return_all_word_tokens
        else:
            return_all_word_tokens = model.return_all_word_tokens

        # ======== compute itc loss ======== #
        # [bs, n_tokens, dim] -> [bs, dim]
        txt_feat = (
            outputs.txt_feat
            if not return_all_word_tokens
            else outputs.txt_feat[:, 0]
        )
        # for new training data
        t2i_sim_scaled, i2t_sim_scaled, t2i_label, i2t_label, loss_itc = (
            self.info_nce_fn(
                txt_feat=txt_feat,
                img_feat=outputs.img_feat,
                label_t2i=batch["label_t2i"],
                img_ids=batch["image_ids"],
                txt_ids=batch["text_ids"],
                return_sims=True,
            )
        )

        # ======== compute ITC gen loss ======== #
        # 生成キャプションは画像ごとに別物なので、クエリの id ではなく画像で数える。
        # 詳細は pair_generated_captions を参照。
        loss_itc_gen = torch.tensor(0.0, device=outputs.img_feat.device, dtype=dtype)
        if (self.w_itc_gen_loss > 0) and (outputs.gen_txt_feat.dim() > 1):
            n_caps = outputs.gen_txt_feat.shape[1]
            gen_text_feats: torch.Tensor = (
                inputs["batch"]["gen_txt_feat"]
                if self.use_prev_txt_feat
                else outputs.gen_txt_feat
            )

            # [bs, n_caps, n_tokens, dim] -> [bs, n_caps, dim]
            gen_text_feats = (
                gen_text_feats[:, :, 0] if return_all_word_tokens else gen_text_feats
            )

            gen_txt_ids, gen_label = self.pair_generated_captions(
                image_ids=batch["image_ids"],
                text_ids=batch["text_ids"],
                label_t2i=batch["label_t2i"],
                per_image=self.itc_gen_per_image,
            )

            for i in range(n_caps):
                _, _, _, _, loss_itc_gen_i = self.info_nce_fn(
                    txt_feat=gen_text_feats[:, i],
                    img_feat=outputs.img_feat,
                    label_t2i=gen_label,
                    img_ids=batch["image_ids"],
                    txt_ids=gen_txt_ids,
                    return_sims=True,
                )
                loss_itc_gen += loss_itc_gen_i
            loss_itc_gen /= n_caps

        # ======== compute ITC dense loss ======== #
        loss_itc_dense = torch.tensor(
            0.0, device=outputs.img_feat.device, dtype=dtype
        )
        if (self.w_itc_dense_loss > 0) and (outputs.dense_txt_feats is not None):
            n_caps = outputs.dense_txt_feats.shape[1]
            dense_text_feats = outputs.dense_txt_feats

            # [bs, n_caps, n_tokens, dim] -> [bs, n_caps, dim]
            dense_text_feats = (
                dense_text_feats[:, :, 0]
                if return_all_word_tokens
                else dense_text_feats
            )

            if dense_text_feats.dim() == 4:  # noqa: PLR2004 (4D tensor)
                dense_text_feats = dense_text_feats[:, :, 0]

            for i in range(n_caps):
                _, _, _, _, loss_itc_dense_i = self.info_nce_fn(
                    txt_feat=dense_text_feats[:, i],
                    img_feat=outputs.img_feat,
                    label_t2i=batch["label_t2i"],
                    img_ids=batch["image_ids"],
                    txt_ids=batch["text_ids"],
                    return_sims=True,
                )
                loss_itc_dense += loss_itc_dense_i
            loss_itc_dense /= n_caps

        return (
            t2i_sim_scaled,
            i2t_sim_scaled,
            t2i_label,
            i2t_label,
            loss_itc,
            loss_itc_gen,
            loss_itc_dense,
        )

    def compute_lm_losses(
        self, outputs: ModelOutput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = outputs.img_feat.dtype
        # ======== compute lm loss ======== #
        # new caption
        new_lm_loss = torch.tensor(0.0, device=outputs.img_feat.device, dtype=dtype)

        if outputs.new_text_logits.shape[0] > 0:
            new_lm_loss = self.lm_loss_fn.compute(
                outputs.new_text_logits, outputs.new_text_label
            )

        # gen caption
        gen_lm_loss = torch.tensor(0.0, device=outputs.img_feat.device, dtype=dtype)
        if (
            outputs.gen_text_logits is not None
            and outputs.gen_text_label is not None
        ) and self.w_cap_loss > 0:
            gen_lm_loss = self.lm_loss_gen_fn.compute(
                outputs.gen_text_logits, outputs.gen_text_label
            )

        # dense caption
        dense_lm_loss = torch.tensor(
            0.0, device=outputs.img_feat.device, dtype=dtype
        )
        if (
            outputs.densecap_text_logits is not None
            and outputs.densecap_text_label is not None
        ):
            dense_lm_loss = self.lm_loss_gen_fn.compute(
                outputs.densecap_text_logits, outputs.densecap_text_label
            )

        return new_lm_loss, gen_lm_loss, dense_lm_loss

    def compute_gen_distill_loss(
        self,
        model: BaseModel,
        outputs: ModelOutput,
        batch: BatchImgTxtOutput,
        t2i_label: torch.Tensor,
        i2t_label: torch.Tensor,
        inputs: dict,
    ) -> torch.Tensor:
        dtype = outputs.img_feat.dtype

        if isinstance(model, DistributedDataParallel):
            return_all_word_tokens = model.module.return_all_word_tokens

        else:
            return_all_word_tokens = model.return_all_word_tokens

        # ========== compute gen sim distill loss ======== #
        loss_gen_sim_distill = torch.tensor(
            0.0, device=outputs.img_feat.device, dtype=dtype
        )

        # need for evaluation phase
        has_prev_feats = (
            batch["prev_img_tensor"] is not None
            and batch["prev_text_tensor"] is not None
        )

        if not ((self.loss_gen_sim_distill_fn is not None) and has_prev_feats):
            return loss_gen_sim_distill

        # num generated captions
        n_caps = outputs.gen_txt_feat.shape[1]

        # [bs, n_caps, n_tokens, dim] -> [bs, n_caps, dim]
        gen_txt_feat = (
            outputs.gen_txt_feat[:, :, 0]
            if return_all_word_tokens
            else outputs.gen_txt_feat
        )

        prev_img_tensor = batch["prev_img_tensor"]
        # Target sims: the prev model.
        if prev_img_tensor is None:
            msg = "prev_img_tensor is None"
            raise ValueError(msg)

        for i in range(n_caps):
            # get organized sim on prev img feature and prev gen txt feature
            _, _, gen_prev_t2i_sim, gen_prev_i2t_sim = Retriever.run(
                txt_feat=inputs["batch"]["gen_txt_feat"][:, i],
                img_feat=prev_img_tensor,
                label_t2i=batch["label_t2i"],
                img_ids=batch["image_ids"],
                txt_ids=batch["text_ids"],
            )

            # get organized sim on current img feature and gen txt feature
            _, _, gen_curr_t2i_sim, gen_curr_i2t_sim = Retriever.run(
                txt_feat=gen_txt_feat[:, i],
                img_feat=outputs.img_feat,
                label_t2i=batch["label_t2i"],
                img_ids=batch["image_ids"],
                txt_ids=batch["text_ids"],
            )

            # remove duplicates in the prev models' features.
            _, gen_prev_txt_feat_wo_dup, prev_img_feat_wo_dup = (
                Retriever.orgnize_prediction(
                    inputs["batch"]["gen_txt_feat"],
                    prev_img_tensor,
                    batch["label_t2i"],
                    batch["text_ids"],
                    batch["image_ids"],
                )
            )

            # remove duplicates in the current models' features.
            _, gen_curr_txt_feat_wo_dup, curr_img_feat_wo_dup = (
                Retriever.orgnize_prediction(
                    gen_txt_feat,
                    outputs.img_feat,
                    batch["label_t2i"],
                    batch["text_ids"],
                    batch["image_ids"],
                )
            )
            distill_input = DistillInput(
                t2i_sim=self.info_nce_fn.scale_sim(gen_curr_t2i_sim),
                i2t_sim=self.info_nce_fn.scale_sim(gen_curr_i2t_sim),
                prev_t2i_sim=self.info_nce_fn.scale_sim(gen_prev_t2i_sim.detach()),
                prev_i2t_sim=self.info_nce_fn.scale_sim(gen_prev_i2t_sim.detach()),
                t2i_label=t2i_label,
                i2t_label=i2t_label,
                img_feats=curr_img_feat_wo_dup,
                txt_feats=gen_curr_txt_feat_wo_dup,
                prev_img_feats=prev_img_feat_wo_dup,
                prev_txt_feats=gen_prev_txt_feat_wo_dup,
            )

            loss_gen_sim_distill += (
                self.loss_gen_sim_distill_fn.compute(
                    distill_input=distill_input,
                )
                / n_caps
            )
        return loss_gen_sim_distill

    def compute_loss(
        self,
        model: BaseModel,
        inputs: dict[str, BatchImgTxtOutput],
        return_outputs: bool = False,
        num_items_in_batch: None | int = None,  # noqa:ARG002
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        global_step = self.state.global_step

        extract_gencap_feat = bool(
            self.w_itc_gen_loss > 0 or self.loss_gen_sim_distill_fn is not None
        )
        outputs: ModelOutput = model(
            **inputs, extract_gencap_feat=extract_gencap_feat
        )

        dtype = outputs.img_feat.dtype
        batch = inputs["batch"]

        # ======== compute itc loss ======== #
        (
            _t2i_sim_scaled,
            _i2t_sim_scaled,
            t2i_label,
            i2t_label,
            loss_itc,
            loss_itc_gen,
            loss_itc_dense,
        ) = self.compute_itc_losses(
            outputs=outputs, batch=batch, model=model, inputs=inputs
        )

        # ======== compute lm loss ======== #
        new_lm_loss, gen_lm_loss, dense_lm_loss = self.compute_lm_losses(outputs)

        # ========== compute gen distill loss ======== #
        loss_gen_sim_distill = self.compute_gen_distill_loss(
            model=model,
            outputs=outputs,
            batch=batch,
            t2i_label=t2i_label,
            i2t_label=i2t_label,
            inputs=inputs,
        )

        w_cap_loss = self.w_cap_loss.to(dtype).to(outputs.img_feat.device)
        w_itc_gen_loss = self.w_itc_gen_loss.to(dtype).to(outputs.img_feat.device)
        w_itc_dense_loss = self.w_itc_dense_loss.to(dtype).to(
            outputs.img_feat.device
        )
        w_dense_cap_loss = self.w_dense_cap_loss.to(dtype).to(
            outputs.img_feat.device
        )
        w_itc_loss = self.w_itc_loss.to(dtype).to(outputs.img_feat.device)

        loss = (
            w_itc_loss * loss_itc
            + new_lm_loss
            + w_cap_loss * gen_lm_loss
            + w_itc_gen_loss * loss_itc_gen
            + w_itc_dense_loss * loss_itc_dense
            + loss_gen_sim_distill
            + w_dense_cap_loss * dense_lm_loss
        )

        is_main_process = self.args.local_rank in [-1, 0]
        is_logging_iter = global_step % self.args.logging_steps == 0

        trainer_output = TrainerOutput(
            txt_feat=outputs.txt_feat,
            img_feat=outputs.img_feat,
            caption_tokens=outputs.caption_tokens,
            dataset_names=outputs.dataset_names,
            gen_txt_feat=outputs.gen_txt_feat,
            loss=loss,
            loss_itc=loss_itc,
            loss_itc_gen=loss_itc_gen,
            loss_new_lm=new_lm_loss,
            loss_gen_lm=gen_lm_loss,
            loss_gen_sim_distill=loss_gen_sim_distill,
            loss_dense_cap=dense_lm_loss,
            loss_itc_dense=loss_itc_dense,
        )

        if not is_main_process or not is_logging_iter:
            # distributed training. Only first process will log metrics
            return (loss, trainer_output) if return_outputs else loss

        if is_logging_iter:
            dataset_name = inputs["batch"].get("dataset_names", ["unknown"])[0]
            # Use .detach().item() to avoid keeping computation graph
            loss_dict = {
                f"{dataset_name}/{k}": v.detach().item()
                if isinstance(v, torch.Tensor)
                else v
                for k, v in trainer_output.items()
                if "loss" in k
            }

            self.log(loss_dict)

        return (loss, trainer_output) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics(
                    "train", start_time, num_tokens=self.state.num_input_tokens_seen
                )

        output = {**logs, "step": self.state.global_step + self.accum_train_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args,
            self.state,
            self.control,  # type: ignore[has-type]
            logs,
        )


class HFTrainerBuilder:
    @staticmethod
    def build(
        train_dataset: ImgTxtDataset,
        eval_datasets: dict[str, ImgTxtDataset],
        model: BaseModel | BasePeftLoraModel,
        model_builder: ModelBuilder,
        pad_token_id: int,
        cfg: CfgArgParser,
        run_logger: LocalLogger | None = None,
        accum_train_step: int = 0,
        train_only: bool = False,
        seed: int = 256,
    ) -> tuple[HFTrainer, transformers.TrainingArguments]:
        torch.set_num_threads(1)
        # calc # of iteration, eval_step
        arguments = cfg.to_train_args()

        optim_target_modules = (
            None if isinstance(model, PeftModel) else model_builder.target_modules
        )
        save_safetensors = isinstance(model, PeftModel)

        accelerate_config = AcceleratorConfig(
            dispatch_batches=False, non_blocking=True, split_batches=False
        ).to_dict()

        trainer_args = transformers.TrainingArguments(
            output_dir=cfg.task_config.output_dir,
            auto_find_batch_size=False,
            optim_target_modules=optim_target_modules,
            fp16=False,
            bf16_full_eval=False,
            half_precision_backend="auto",
            gradient_checkpointing=False,
            logging_strategy="steps",
            logging_first_step=False,
            report_to=["tensorboard"],
            eval_strategy="steps",
            eval_on_start=False,
            eval_accumulation_steps=1,
            save_strategy="steps",
            save_steps=0.05,
            save_safetensors=save_safetensors,
            save_total_limit=1,
            seed=seed,
            data_seed=seed,
            ddp_find_unused_parameters=True,
            dataloader_num_workers=2,
            dataloader_prefetch_factor=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=False,
            label_names=LABEL_VARS,  # subset of dataset.BatchImgTxtOutput
            accelerator_config=accelerate_config,
            ddp_timeout=600,
            overwrite_output_dir=True,
            **arguments,
        )

        if "_task" not in cfg.task_config.output_dir:
            output_dir = None
        else:
            output_dir = Path(cfg.task_config.output_dir)

        callbacks: list[Any] = [SyncOnStepEndCallback()]
        if run_logger is not None:
            callbacks.append(LogCallback(run_logger))

        trainer = HFTrainer(
            model=model,
            train_dataset=train_dataset,
            mixin=model_builder.mixin,
            gen_sim_distill=cfg.task_config.gen_sim_distill,
            eval_dataset=eval_datasets,
            compute_metrics=None
            if train_only
            else MetricCalculator(100, output_dir).compute_metrics,
            img_size=model_builder.img_size,
            args=trainer_args,
            data_collator=ImgTxtCollator(pad_token_id),
            callbacks=callbacks,
            w_cap_loss=cfg.task_config.w_cap_loss,
            w_itc_gen_loss=cfg.task_config.w_itc_gen_loss,
            w_itc_dense_loss=cfg.task_config.w_itc_dense_loss,
            accum_train_step=accum_train_step,
            temperature=model.temp,
            use_prev_txt_feat=cfg.task_config.use_prev_txt_feat,
            w_dense_cap_loss=cfg.task_config.w_dense_cap_loss,
            cap_reduction=cfg.task_config.cap_reduction,
            w_itc_loss=cfg.task_config.w_itc_loss,
            itc_gen_per_image=cfg.task_config.itc_gen_per_image,
        )

        return trainer, trainer_args

    @staticmethod
    def build_for_prediction(
        model: BaseModel | BasePeftLoraModel,
        model_builder: ModelBuilder,
        output_dir: str,
        batch_size: int,
        pad_token_id: int,
        compute_metrics: None | Callable = None,
        batch_eval_metrics: bool = True,
        seed: int = 256,
    ) -> tuple[HFTrainer, transformers.TrainingArguments]:
        torch.set_num_threads(1)
        # calc # of iteration, eval_step

        save_safetensors = isinstance(model, PeftModel)

        accelerate_config = AcceleratorConfig(
            dispatch_batches=False, non_blocking=True, split_batches=False
        ).to_dict()

        trainer_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            auto_find_batch_size=False,
            batch_eval_metrics=batch_eval_metrics,
            ddp_find_unused_parameters=True,
            fp16=False,
            bf16=True,
            bf16_full_eval=False,
            half_precision_backend="auto",
            output_dir=output_dir,
            eval_accumulation_steps=1,
            save_safetensors=save_safetensors,
            seed=seed,
            data_seed=seed,
            dataloader_num_workers=2,
            dataloader_prefetch_factor=4,
            dataloader_persistent_workers=False,
            dataloader_pin_memory=True,
            label_names=LABEL_VARS,  # subset of dataset.BatchImgTxtOutput
            accelerator_config=accelerate_config,
            ddp_timeout=600,
            gradient_checkpointing=False,
            overwrite_output_dir=True,
            report_to="none",
            logging_strategy="no",
        )

        trainer = HFTrainer(
            model=model,
            mixin=model_builder.mixin,
            gen_sim_distill="",
            compute_metrics=compute_metrics,
            img_size=model_builder.img_size,
            args=trainer_args,
            data_collator=ImgTxtCollator(pad_token_id),
            callbacks=[SyncOnStepEndCallback()],
            w_cap_loss=-1.0,
            w_itc_gen_loss=-1.0,
            w_itc_dense_loss=-1.0,
            temperature=model.temp,
            use_prev_txt_feat=False,
            w_dense_cap_loss=-1.0,
            w_itc_loss=-1.0,
        )

        return trainer, trainer_args
