import math
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
from lavis.common.registry import registry as lavis_registry
from lavis.models import load_model_and_preprocess
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from lavis.models.blip2_models.Qformer import BertSelfAttention
from lavis.models.eva_vit import Attention as EvaAttentionType
from torch.nn import functional

from capcl.dataset import BatchImgTxtOutput
from capcl.models.model import (
    LORA_SUPPORTED_MODULES,
    BaseMixin,
    BaseModel,
    ModelBuilder,
    ModelOutput,
)
from capcl.models.modules.normalize import (
    l2_normalization,
    wrap_model_layernorm_dtype_consistent,
)
from capcl.registry_class import registry

if TYPE_CHECKING:
    from lavis.models.blip2_models.Qformer import BertLMHeadModel
    from lavis.models.eva_vit import VisionTransformer
    from transformers import PreTrainedTokenizerBase


BLIP2_MODELS = [
    "Salesforce/blip2-itm-vit-g",
    "Salesforce/blip2-itm-vit-g-coco",
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-opt-2.7b-coco",
    "Salesforce/blip2-opt-6.7b",
    "Salesforce/blip2-opt-6.7b-coco",
    "Salesforce/blip2-flan-t5-xxl",
    "Salesforce/blip2-flan-t5-xl",
    "Salesforce/blip2-flan-t5-xl-coco",
]

BLIP2_PRETRAIN_MODELS = ["coco", "pretrain_vitL", "pretrain"]

BLIP2_MODULES = [
    "all",
    "qformer",
    "qformer.bert",
    "qformer.bert.proj",
    "qformer.bert.query",
    "qformer.bert.proj.query",
    "qformer.bert.query.proj",
]

QUERY_NAME = "Qformer.query_tokens_embeds"
VISION_PROJ_NAME = "vision_proj"
TEXT_PROJ_NAME = "text_proj"

WORD_EMBEDS_NAME = "Qformer.bert.embeddings.word_embeddings"
POS_EMBEDS_NAME = "Qformer.bert.embeddings.position_embeddings"
DECODER_NAME = "Qformer.cls.predictions.decoder"
PAD_TOKEN_ID = -100
REL_POS_BIAS_RANK = 3

MASK_MIN_TH = 1e-5


class EvaAttentionWrapper(torch.nn.Module):
    def __init__(self, attn: EvaAttentionType) -> None:
        super().__init__()
        self.attn = attn

    def forward(
        self, x: torch.Tensor, rel_pos_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        attn = self.attn
        bsz, seq_len, _ = x.shape

        qkv_bias = None
        if attn.q_bias is not None:
            q_bias = cast("torch.Tensor", attn.q_bias)
            v_bias = cast("torch.Tensor", attn.v_bias)
            qkv_bias = torch.cat(
                (q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias)
            )

        qkv_flatten = functional.linear(
            input=x, weight=attn.qkv.weight, bias=qkv_bias
        )
        qkv = qkv_flatten.reshape(bsz, seq_len, 3, attn.num_heads, -1).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = attn.scale

        attn_bias = None
        if attn.relative_position_bias_table is not None:
            # bias info
            window_size = cast("tuple[int, int]", attn.window_size)
            relative_position_bias_table = cast(
                "torch.Tensor", attn.relative_position_bias_table
            )
            relative_position_index = cast(
                "torch.Tensor", attn.relative_position_index
            )

            # bias
            relative_position_bias_permuted = relative_position_bias_table[
                relative_position_index.view(-1)
            ].view(
                window_size[0] * window_size[1] + 1,
                window_size[0] * window_size[1] + 1,
                -1,
            )
            relative_position_bias = relative_position_bias_permuted.permute(
                2, 0, 1
            ).contiguous()
            attn_bias = relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            rel_pos = rel_pos_bias
            if rel_pos.dim() == REL_POS_BIAS_RANK:
                rel_pos = rel_pos.unsqueeze(0)
            attn_bias = rel_pos if attn_bias is None else attn_bias + rel_pos

        dropout_p = attn.attn_drop.p if attn.training else 0.0

        attn_out = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )
        attn_out_transposed = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        proj_output = attn.proj(attn_out_transposed)
        return attn.proj_drop(proj_output)


class BertSelfAttentionWrapper(torch.nn.Module):
    def __init__(self, attn: BertSelfAttention) -> None:
        super().__init__()
        self.attn = attn

    def forward(  # noqa: PLR0912,C901
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], ...]:
        attn = self.attn

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = attn.transpose_for_scores(attn.key(encoder_hidden_states))
            value_layer = attn.transpose_for_scores(
                attn.value(encoder_hidden_states)
            )
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = attn.transpose_for_scores(attn.key(hidden_states))
            value_layer = attn.transpose_for_scores(attn.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = attn.transpose_for_scores(attn.key(hidden_states))
            value_layer = attn.transpose_for_scores(attn.value(hidden_states))

        mixed_query_layer = attn.query(hidden_states)
        query_layer = attn.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        rel_pos_bias: torch.Tensor | None = None
        if attn.position_embedding_type in {
            "relative_key",
            "relative_key_query",
        }:
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = attn.distance_embedding(
                distance + attn.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)

            if attn.position_embedding_type == "relative_key":
                rel_pos_bias = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
            else:
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                rel_pos_bias = (
                    relative_position_scores_query + relative_position_scores_key
                )

        attention_probs: torch.Tensor | None = None
        if output_attentions:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            if rel_pos_bias is not None:
                attention_scores = attention_scores + rel_pos_bias
            attention_scores = attention_scores / math.sqrt(attn.attention_head_size)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        if head_mask is not None:
            value_layer = value_layer * head_mask

        if attention_mask is not None and attention_mask.abs().min() > MASK_MIN_TH:
            sdpa_mask = attention_mask
        else:
            sdpa_mask = None

        if rel_pos_bias is not None:
            sdpa_mask = (
                rel_pos_bias if sdpa_mask is None else sdpa_mask + rel_pos_bias
            )

        dropout_p = attn.dropout.p if attn.training else 0.0

        context_layer = functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(
            *context_layer.size()[:-2], attn.all_head_size
        )

        outputs = (
            (context_layer, attention_probs)
            if output_attentions
            else (context_layer,)
        )
        return (*outputs, past_key_value)  # pyright: ignore[reportReturnType] # ty: ignore[invalid-return-type]


@lavis_registry.register_model("blip2_pretrain")
class Blip2Pretrain(Blip2Qformer, BaseModel):
    def type_definition_for_pyright(self) -> None:
        self.query_tokens: torch.Tensor
        self.Qformer: BertLMHeadModel
        self.tokenizer: PreTrainedTokenizerBase
        self.text_proj: torch.nn.Linear
        self.vision_proj: torch.nn.Linear
        self.ln_vision: torch.nn.LayerNorm
        self.visual_encoder: VisionTransformer
        self.top_p: float
        self.top_k: int
        self.temperature: float

        self.max_txt_len: int
        self.N_GEN_CAPS: int

    def get_query_tokens(self) -> torch.Tensor:
        n_tokens = self.query_tokens.shape[1]
        device = self.query_tokens.device
        _input = torch.arange(0, n_tokens, device=device).unsqueeze(0)
        embeddings = cast("torch.nn.Embedding", self.Qformer.query_tokens_embeds)
        queries: torch.Tensor = embeddings(_input)

        return queries

    def next_token_prediction(
        self,
        cap_target_texts: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the loss for the captioning task.

        Args:
            cap_target_texts (torch.Tensor): The target texts for captioning. shape [n_bs, n_caps, max_txt_len]
            past_key_values (tuple[tuple[torch.Tensor, torch.Tensor]]): The past key values for the Qformer.
                len(past_key_values) == n_bs, past_key_values[0].shape == [n_bs, n_heads, n_layers, n_heads, n_embd // n_heads]
            encoder_hidden_states (None | torch.Tensor): The encoder hidden states. If provided, used for cluster labels.


        Returns:
            torch.Tensor: predicted logits. shape [n_bs, n_caps, max_txt_len, vocab_size]
            torch.Tensor: labels for the loss function. shape [n_bs, n_caps, max_txt_len]
        """
        if cap_target_texts.dim() == 2:  # noqa: PLR2004
            cap_target_texts = cap_target_texts.unsqueeze(1)

        n_bs = len(cap_target_texts)
        n_caps = cap_target_texts.shape[1]

        device = cap_target_texts.device

        all_text_tokens = cap_target_texts.to(device=device)

        # text_token.shape

        # [n_bs, n_caps, max_txt_len] --> [n_bs * n_caps, max_txt_len]
        text_tokens = (
            all_text_tokens.contiguous().view(n_bs * n_caps, -1).to(device=device)
        )

        org_attention_mask = torch.zeros_like(text_tokens)
        org_attention_mask = org_attention_mask.masked_fill(text_tokens > 0, 1)

        bos_token_id: int = self.tokenizer.bos_token_id  # pyright: ignore[reportAssignmentType]
        pad_token_id: int = self.tokenizer.pad_token_id  # pyright: ignore[reportAssignmentType]

        decoder_input_ids = text_tokens.clone()
        decoder_input_ids[:, 0] = bos_token_id
        labels = torch.masked_fill(
            decoder_input_ids,
            decoder_input_ids == pad_token_id,
            PAD_TOKEN_ID,
        )

        # len(past_key_values) == n_bs --> len(_past_key_values == n_bs * n_caps)
        _past_key_vals: tuple | None = tuple(
            [
                (
                    i[0]
                    .unsqueeze(1)
                    .expand(-1, n_caps, *i[0].shape[1:])
                    .contiguous()
                    .reshape(n_bs * n_caps, *i[0].shape[1:]),
                    i[1]
                    .unsqueeze(1)
                    .expand(-1, n_caps, *i[1].shape[1:])
                    .contiguous()
                    .reshape(n_bs * n_caps, *i[1].shape[1:]),
                )
                for i in past_key_values
            ]
        )

        query_tokens = (
            self.get_query_tokens().expand(text_tokens.shape[0], -1, -1).contiguous()
        )

        query_atts = torch.ones(
            query_tokens.size()[:-1], dtype=torch.long, device=device
        )
        attention_mask: torch.Tensor | None = torch.cat(
            [query_atts, org_attention_mask], dim=1
        )

        # encoder_attention_mask
        encoder_attention_mask = None
        encoder_hidden_states_expanded = None

        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states_expanded,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=_past_key_vals,
            return_dict=True,
        )

        logits = lm_output.logits

        return logits.view(
            n_bs, n_caps, -1, logits.size(-1)
        ).contiguous(), labels.view(n_bs, n_caps, -1).contiguous()

    def extract_text_feats(
        self,
        cap_target_texts: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_bs, n_caps = cap_target_texts.shape[:2]
        input_ids = cap_target_texts.to(device).reshape(
            -1, cap_target_texts.shape[-1]
        )
        # n_bs, n_caps, max_txt_len
        attention_mask = torch.zeros_like(input_ids)
        attention_mask.masked_fill_(input_ids > 0, 1)

        text_output = self.Qformer.bert(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        if self.return_all_word_tokens:
            hidden_state = text_output.last_hidden_state
        else:
            hidden_state = text_output.last_hidden_state[:, 0, :]

        projected: torch.Tensor = self.text_proj(hidden_state)
        text_feat = l2_normalization(projected)

        if self.return_all_word_tokens:
            # fill 0 for the pad tokens
            # text_feat shape: [bs * n_caps, n_tokens, dim]
            # input_ids shape: [bs * n_caps, n_tokens]
            text_feat[input_ids <= 0] = 0.0
            text_feat[input_ids == self.tokenizer.sep_token_id] = 0.0
            text_feat = text_feat.reshape((n_bs, n_caps, text_feat.shape[1], -1))
            return (input_ids, text_feat)
        return input_ids, text_feat.reshape((n_bs, n_caps, -1))

    def forward(  # ty: ignore[invalid-method-override]
        self,
        batch: BatchImgTxtOutput,
        extract_gencap_feat: bool = False,
        **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> ModelOutput:
        """Forward without ITM. Return with features."""
        # TODO(dev): move extract_gencap_feat to the constructor.
        # label variables are passed as kwargs. see transformers.trainer.py line 4432. note that label variables are in the batch.

        image = batch["img_tensor"]
        itc_target_text = batch["new_texts"]
        cap_target_texts = batch["gen_caption"]

        dataset_names = batch["dataset_names"]

        # ====== Image Feature Extraction ======
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image.device
        )

        query_tokens = (
            self.get_query_tokens()
            .expand(image_embeds.shape[0], -1, -1)
            .contiguous()
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = l2_normalization(
            self.vision_proj(query_output.last_hidden_state)
        )

        # ====== Text Feature Extraction ======
        input_ids, _text_feat = self.extract_text_feats(
            itc_target_text, image.device
        )
        text_feat = _text_feat[:, 0]  # n_caps --> 1

        ###==============  ===================###
        gen_txt_feat = torch.tensor([], device=image.device)
        if cap_target_texts.dim() > 1 and extract_gencap_feat:
            _, gen_txt_feat = self.extract_text_feats(
                cap_target_texts=cap_target_texts, device=image.device
            )
            gen_txt_feat = gen_txt_feat.to(torch.bfloat16)

        ##================= Next Token Prediction ========================##

        new_text_logits, new_text_label = self.next_token_prediction(
            cap_target_texts=input_ids,
            past_key_values=query_output.past_key_values,
        )

        gen_text_logits, gen_text_label = None, None
        if cap_target_texts.dim() > 1:
            gen_text_logits, gen_text_label = self.next_token_prediction(
                cap_target_texts=cap_target_texts,
                past_key_values=query_output.past_key_values,
            )

        densecap_text_logits, densecap_text_label = None, None
        if batch["dense_cap"].dim() > 1:
            densecap_text_logits, densecap_text_label = self.next_token_prediction(
                cap_target_texts=batch["dense_cap"],
                past_key_values=query_output.past_key_values,
            )

        ##
        dataset_name_tokens = self.tokenizer(
            dataset_names,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        ## ========================
        if self.training or (not self.training and self.N_GEN_CAPS <= 0):
            return ModelOutput(
                img_feat=image_feats.to(torch.bfloat16),
                txt_feat=text_feat.to(torch.bfloat16),
                gen_txt_feat=gen_txt_feat,
                dataset_names=dataset_name_tokens.input_ids.to(torch.int64),
                caption_tokens=torch.tensor([], device=image.device),
                new_text_label=new_text_label,
                new_text_logits=new_text_logits,
                gen_text_label=gen_text_label,
                gen_text_logits=gen_text_logits,
                densecap_text_label=densecap_text_label,
                densecap_text_logits=densecap_text_logits,
            )

        ## ========================

        # generate captions
        batch_caps, _ = self.image_captioning(
            image,
            self.N_GEN_CAPS,
            image_embeds,
            past_key_values=query_output.past_key_values,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
        )

        gen_txt_feat_flatten = self.extract_text_feats(
            cap_target_texts=batch_caps.view((-1, batch_caps.shape[-1])),
            device=image.device,
        )[1]

        if self.return_all_word_tokens:
            gen_txt_feat = gen_txt_feat_flatten.view(
                image.shape[0], self.N_GEN_CAPS, gen_txt_feat_flatten.shape[1], -1
            )
        else:
            gen_txt_feat = (
                gen_txt_feat_flatten.view(image.shape[0], self.N_GEN_CAPS, -1)
                .contiguous()
                .to(torch.bfloat16)
            )

        return ModelOutput(
            img_feat=image_feats.to(torch.bfloat16),
            txt_feat=text_feat.to(torch.bfloat16),
            gen_txt_feat=gen_txt_feat,
            caption_tokens=batch_caps.to(torch.int64),  # .to(torch.bfloat16),
            dataset_names=dataset_name_tokens.input_ids.to(torch.int64),
            new_text_label=new_text_label,
            new_text_logits=new_text_logits,
            gen_text_label=gen_text_label,
            gen_text_logits=gen_text_logits,
            densecap_text_label=densecap_text_label,
            densecap_text_logits=densecap_text_logits,
        )

    def image_captioning(
        self,
        img_tensor: torch.Tensor,
        n_captions: int = 1,
        image_embeds: None | torch.Tensor = None,
        top_p: float = 0.9,
        top_k: int = 0,
        temperature: float = 1.0,
        past_key_values: None | tuple[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ======= from BLIP2 generation to avoid same forward call =========
        max_length = 128
        use_nucleus_sampling = True
        num_beams = 1
        min_length = 10

        if (top_p > 0.0 and top_p < 1.0) and top_k > 0:
            # both top_p and top_k are set
            _top_p = 1.0
            _top_k = top_k

        elif top_k > 0:
            # only top_k is set
            _top_k = top_k
            _top_p = 1.0
        else:
            # only top_p is set or both are not set
            _top_p = top_p
            _top_k = None

        image = img_tensor

        if image_embeds is None:
            if image.dim() == 3:  # noqa: PLR2004
                image = image.unsqueeze(0)

            image_embeds_pt = self.ln_vision(self.visual_encoder(image))
        else:
            image_embeds_pt = image_embeds

        image_atts = torch.ones(
            image_embeds_pt.size()[:-1], dtype=torch.long, device=image.device
        )

        input_ids = torch.zeros(
            (image.size(0), 1), dtype=torch.int64, device=image.device
        )
        input_ids[:, 0] = self.tokenizer.bos_token_id  # pyright: ignore[reportArgumentType]

        query_tokens = (
            self.get_query_tokens()
            .expand(image_embeds_pt.shape[0], -1, -1)
            .contiguous()
        )

        # ======= generate captions with nucleus sampling =======
        tokens = torch.zeros(
            (image.size(0), n_captions, max_length),
            dtype=torch.int64,
            device=image.device,
        )
        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=_top_p,
            top_k=_top_k,
            temperature=temperature,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            encoder_hidden_states=image_embeds_pt,
            encoder_attention_mask=image_atts,
            num_return_sequences=n_captions,
            past_key_values=past_key_values,
        )
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs[0]
        tokens[:, :, : outputs.shape[1]] = (
            outputs.view(image.size(0), n_captions, -1).to(torch.int64).contiguous()
        )

        return tokens.contiguous(), torch.tensor([], device=image.device)

    def to_bfloat16(self) -> None:
        self.to(torch.bfloat16)
        self.get_query_tokens().to(torch.bfloat16)
        self.Qformer.to(torch.bfloat16)  # pyright: ignore[reportArgumentType] # ty: ignore[invalid-argument-type]
        self.Qformer.bert.embeddings = self.Qformer.bert.embeddings.to(
            torch.bfloat16
        )


@registry.register_model_builder("blip2_pretrain")
class BLIP2PretrainBuilder(ModelBuilder):
    def __init__(
        self,
        model_class: str,
        model_type: str,
        top_p: float,
        top_k: int,
        temperature: float,
        target_module_name: str = "qformer.bert",
        sdpa_backend: Literal["org", "torch"] = "org",
    ) -> None:
        self._has_lm_head = True

        if model_type not in BLIP2_PRETRAIN_MODELS:
            msg = f"model_type must be one of {BLIP2_PRETRAIN_MODELS}. Got {model_type}"
            raise ValueError(msg)

        if target_module_name not in BLIP2_MODULES:
            msg = f"target_module_name must be one of {BLIP2_MODULES}. Got {target_module_name}"
            raise ValueError(msg)

        self._model_class = model_class
        self._model_type = model_type

        _model, img_processor, txt_processor = load_model_and_preprocess(
            "blip2_pretrain", model_type, is_eval=True
        )

        self._model: Blip2Pretrain = _model  # pyright: ignore[reportIncompatibleVariableOverride]

        self._model.top_p = top_p  # pyright: ignore[reportArgumentType]
        self._model.top_k = top_k  # pyright: ignore[reportArgumentType]
        self._model.temperature = temperature  # pyright: ignore[reportArgumentType]

        query_tokens: torch.Tensor = self._model.query_tokens  # pyright: ignore[reportAssignmentType]
        _, n_query, query_dim = query_tokens.shape

        query_embeds = torch.nn.Embedding(
            num_embeddings=n_query, embedding_dim=query_dim
        )
        query_embeds.weight.data.copy_(query_tokens.squeeze().data)
        self._model.Qformer.query_tokens_embeds = query_embeds  # pyright: ignore[reportAttributeAccessIssue]

        if sdpa_backend == "torch":
            self._wrap_visual_attention_by_torch_native()
            self._wrap_qformer_attention_by_torch_native()

        self._model.to(torch.bfloat16)
        self._model.to_bfloat16()  # pyright: ignore[reportCallIssue]

        # wrap LayerNorm to be dtype-consistent
        wrap_model_layernorm_dtype_consistent(self._model)

        if hasattr(self._model.visual_encoder, "use_checkpoint"):
            # BLIP2 modules use two types of ViT; models.eva_vit.VisionTransformer and models.clip_vit.VisionTransformer.
            # eva_vit uses activation checkpointing, while clip_vit uses gradient checkpointing.
            # To work with DDP, I modiefied the eva_vits' forward pass to force reentrant=False in checkpointing.
            # To work with DDP, I permitted the gradient checkpointing in clip_vit.

            # eva_vit: activation checkpointing is enabled but reentrant=False in the forward pass.
            self._model.visual_encoder.use_checkpoint = False  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]

        if img_processor is None or txt_processor is None:
            msg = "img_processor and txt_processor must not be None"
            raise ValueError(msg)

        self._train_img_processor = img_processor["train"]
        self._eval_img_processor = img_processor["eval"]
        self._txt_processor = txt_processor["train"]

        self._txt_processor.max_len = 128
        self._model.tokenizer.max_len = self._txt_processor.max_len  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]

        self._img_size = img_processor["train"].transform.transforms[0].size

        self._target_modules = self.get_target_modules(target_module_name)

        self._mixin = BaseMixin()
        self._mixin.img_size = self._img_size
        self._target_module_name = target_module_name

    def _wrap_visual_attention_by_torch_native(self) -> None:
        blocks = self._model.visual_encoder.blocks

        for block in blocks:  # pyright: ignore[reportGeneralTypeIssues]
            if not hasattr(block, "attn"):
                continue
            attn = cast("EvaAttentionType", block.attn)
            block.attn = EvaAttentionWrapper(attn)

    def _wrap_qformer_attention_by_torch_native(self) -> None:
        qformer = self._model.Qformer

        if not hasattr(qformer, "bert"):
            return

        layers = qformer.bert.encoder.layer

        for layer in layers:  # pyright: ignore[reportGeneralTypeIssues]
            if hasattr(layer, "attention"):
                attention_module = cast("Any", layer.attention)
                attn_self = cast("BertSelfAttention", attention_module.self)
                attention_module.self = BertSelfAttentionWrapper(attn_self)
            if hasattr(layer, "has_cross_attention") and layer.has_cross_attention:
                cross_attention_module = cast("Any", layer.crossattention)
                cross_attn_self = cast(
                    "BertSelfAttention", cross_attention_module.self
                )
                cross_attention_module.self = BertSelfAttentionWrapper(
                    cross_attn_self
                )

    def _get_bert_encoder_modules(self) -> list[str]:
        target_modules = []

        model: Blip2Qformer = cast("Blip2Qformer", self.model)

        # bert.encoder. **without word embedding**
        for k, v in model.Qformer.bert.encoder.named_modules():  # pyright: ignore[reportAttributeAccessIssue]
            if type(v) in LORA_SUPPORTED_MODULES:
                target_modules.append("Qformer.bert.encoder." + k)
                self.enable_require_grad(v.parameters())

        return target_modules

    def _get_cls_modules(self) -> list[str]:
        # Qformer.cls
        target_modules = []
        model: Blip2Qformer = cast("Blip2Qformer", self.model)
        for k, v in model.Qformer.cls.named_modules():  # pyright: ignore[reportAttributeAccessIssue]
            if type(v) in LORA_SUPPORTED_MODULES:
                target_modules.append("Qformer.cls." + k)
                self.enable_require_grad(v.parameters())

        return target_modules

    def get_target_modules(self, target_module_name: str) -> list[str]:
        """Get target modules.

        Qformer has [bert, cls] modules. bert has [encoder, embeddings] modules.

        target_module_name == qformer -> ["Qformer.cls", "Qformer.bert.encoder"]
        target_module_name == qformer.bert -> ["Qformer.bert.encoder"]
        target_module_name == all -> all modules including [visual_encoder, vision_proj, text_proj, itm_head, Qformer]

        learnable query or model.Qformer.query_tokens is currently out of scope, as it is defined as nn.Parameter, which is not supported by LORA.
        A possible workaround is to wraps the query tokens by nn.Embedding and to redefine the forward method of the model.
        Ref: https://github.com/huggingface/peft/issues/1272

        Args:
            target_module_name (str): _description_

        Returns:
            list[str]: _description_
        """
        self.disable_all_require_grad()

        # -- all modules
        if target_module_name == "all":
            return self.enable_all_modules(
                True  # noqa: FBT003
            )

        # bert.encoder.
        target_modules = set(self._get_bert_encoder_modules())

        if target_module_name == "qformer":
            # Qformer.cls
            target_modules.update(self._get_cls_modules())

        additional_modules = set()

        if target_module_name == "qformer" or "query" in target_module_name:
            # learnable queries
            additional_modules.add(QUERY_NAME)

        if target_module_name == "qformer" or "proj" in target_module_name:
            # text_proj, vision_proj
            additional_modules.add(TEXT_PROJ_NAME)
            additional_modules.add(VISION_PROJ_NAME)

        if target_module_name == "qformer":
            # word embeddings and their positional embeddings.
            additional_modules.add(WORD_EMBEDS_NAME)
            additional_modules.add(POS_EMBEDS_NAME)
            # Qformer.bert.embeddings.word_embeddings is the same as Qformer.cls.predictions.decoder
            # in the BLIP2 model.
            additional_modules.add(DECODER_NAME)

        named_modules = dict(self.model.named_modules())
        for name in additional_modules:
            target_modules.add(name)
            self.enable_require_grad(named_modules[name].parameters())

        return list(target_modules)
