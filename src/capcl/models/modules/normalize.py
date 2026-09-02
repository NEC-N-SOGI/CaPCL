import torch
from lavis.models.blip2_models.blip2 import LayerNorm as LavisLayerNorm
from torch.nn import functional


def l2_normalization(x: torch.Tensor) -> torch.Tensor:
    """L2 Normalization along the last dimension with eps for numerical stability."""
    denom = (
        x.norm(2, dim=-1, keepdim=True, dtype=x.dtype).clamp(min=1e-10).expand_as(x)
    )
    normalized = x / denom

    if not isinstance(normalized, torch.Tensor):
        msg = ""
        raise TypeError(msg)

    return normalized


class DtypeConsistentLayerNorm(torch.nn.Module):
    def __init__(
        self,
        normalized_shape: tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = weight
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype

        normed = functional.layer_norm(
            x,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )

        return normed.to(orig_type)


def _get_parent_module(
    model: torch.nn.Module, module_name: str
) -> tuple[torch.nn.Module, str]:
    """Get parent module and attribute name of the given module name.

    Args:
        model (torch.nn.Module): The model.
        module_name (str): The module name.

    Returns:
        tuple[torch.nn.Module, str]: The parent module and attribute name.
    """
    names = module_name.split(".")
    parent_module = model
    for name in names[:-1]:
        parent_module = getattr(parent_module, name)
    attr_name = names[-1]
    return parent_module, attr_name


def wrap_model_layernorm_dtype_consistent(model: torch.nn.Module) -> list[str]:
    wrapped_modules = []
    # wrap LayerNorm to be dtype-consistent
    # if we use transformers, convert_fp32 will be applied after LayerNorm.
    # Lavis's LayerNorm explicitly cast fp32.

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.LayerNorm, LavisLayerNorm)):
            dtype_consistent_ln = DtypeConsistentLayerNorm(
                module.normalized_shape,  # pyright: ignore[reportArgumentType]
                module.eps,
                module.elementwise_affine,
                module.weight,
                module.bias,
            )
            parent_module, attr_name = _get_parent_module(model, name)
            setattr(parent_module, attr_name, dtype_consistent_ln)
            wrapped_modules.append(name)
    return wrapped_modules
