from pydantic import BaseModel, Field, StrictFloat


class EvalMetrics(BaseModel):
    n_imgs: int = Field(0, description="Number of images")
    n_txts: int = Field(0, description="Number of texts")
    t2i_r1: StrictFloat = Field(
        0.0, description="Recall@1 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_r5: StrictFloat = Field(
        0.0, description="Recall@5 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_r10: StrictFloat = Field(
        0.0, description="Recall@10 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_map_20: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_map_40: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_map_60: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_map_80: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_map_100: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_map_200: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_map_400: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_map_600: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_map_800: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_map_1000: StrictFloat = Field(
        0.0, description="Mean Average Precision for text-to-image retrieval"
    )
    t2i_pk_20: StrictFloat = Field(
        0.0, description="Precision@20 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_pk_40: StrictFloat = Field(
        0.0, description="Precision@40 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_pk_60: StrictFloat = Field(
        0.0, description="Precision@60 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_pk_80: StrictFloat = Field(
        0.0, description="Precision@80 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_pk_100: StrictFloat = Field(
        0.0, description="Precision@100 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_pk_200: StrictFloat = Field(
        0.0, description="Precision@200 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_pk_400: StrictFloat = Field(
        0.0, description="Precision@400 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_pk_600: StrictFloat = Field(
        0.0, description="Precision@600 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_pk_800: StrictFloat = Field(
        0.0, description="Precision@800 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_pk_1000: StrictFloat = Field(
        0.0, description="Precision@1000 for text-to-image retrieval", ge=0.0, le=1.0
    )
    t2i_mrr: StrictFloat = Field(
        0.0, description="Mean Reciprocal Rank for text-to-image retrieval"
    )
    i2t_r1: StrictFloat = Field(
        0.0, description="Recall@1 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_r5: StrictFloat = Field(
        0.0, description="Recall@5 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_r10: StrictFloat = Field(
        0.0, description="Recall@10 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_mrr: StrictFloat = Field(
        0.0, description="Mean Reciprocal Rank for image-to-text retrieval"
    )
    i2t_map_20: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_map_40: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_map_60: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_map_80: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_map_100: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_map_200: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_map_400: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_map_600: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_map_800: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_map_1000: StrictFloat = Field(
        0.0, description="Mean Average Precision for image-to-text retrieval"
    )
    i2t_pk_20: StrictFloat = Field(
        0.0, description="Precision@20 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_pk_40: StrictFloat = Field(
        0.0, description="Precision@40 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_pk_60: StrictFloat = Field(
        0.0, description="Precision@60 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_pk_80: StrictFloat = Field(
        0.0, description="Precision@80 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_pk_100: StrictFloat = Field(
        0.0, description="Precision@100 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_pk_200: StrictFloat = Field(
        0.0, description="Precision@200 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_pk_400: StrictFloat = Field(
        0.0, description="Precision@400 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_pk_600: StrictFloat = Field(
        0.0, description="Precision@600 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_pk_800: StrictFloat = Field(
        0.0, description="Precision@800 for image-to-text retrieval", ge=0.0, le=1.0
    )
    i2t_pk_1000: StrictFloat = Field(
        0.0, description="Precision@1000 for image-to-text retrieval", ge=0.0, le=1.0
    )
