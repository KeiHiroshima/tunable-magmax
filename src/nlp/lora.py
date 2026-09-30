"""LoRA for --finetune_mode lora (t5-large), following O-LoRA's T5 setting:
r=8, alpha=32, dropout 0.1 on the q and v projections of every T5 attention
block (encoder self-attention, decoder self- and cross-attention) — peft's
default target modules for T5, which O-LoRA uses unchanged.

Written here rather than taken from `peft` because the pipeline needs only
the four operations below, and needs them explicit: continual learning
merges each task's adapter into the weights before the next task starts
(O-LoRA eq. 9), and the merge step rebuilds each task's model from the
stored adapters alone.

Only the adapter is saved per task. After task t the model is

    W_t = W_0 + sum_{i<=t} (alpha / r) * B_i A_i

on the adapted weights and W_0 everywhere else, so its task vector is the
sum of the adapters' deltas (adapters_to_vector).
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LoraConfig:
    r: int = 8
    alpha: int = 32
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("q", "v")


class LoRALinear(nn.Module):
    """base(x) + (alpha / r) * B A dropout(x), with base frozen. B starts at
    zero, so an injected model computes exactly what the base model did."""

    def __init__(self, base: nn.Linear, config: LoraConfig):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(base.weight.new_zeros(config.r, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, config.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = config.alpha / config.r
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return self.base(x) + lora * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


def _target_linears(model: nn.Module, config: LoraConfig) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in config.target_modules
    ]


def inject_lora(model: nn.Module, config: LoraConfig) -> list[str]:
    """Freeze every existing parameter and wrap each target nn.Linear in a
    LoRALinear, whose A/B are then the only trainable parameters. Returns the
    wrapped modules' names."""
    for p in model.parameters():
        p.requires_grad = False
    names = _target_linears(model, config)
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), config))
    return names


def extract_adapter(model: nn.Module, config: LoraConfig) -> dict:
    """The adapter to save after a task: every LoRALinear's A and B, keyed by
    module name, plus the config needed to scale them back."""
    modules = {
        name: {"A": m.lora_A.detach().cpu().clone(), "B": m.lora_B.detach().cpu().clone()}
        for name, m in model.named_modules()
        if isinstance(m, LoRALinear)
    }
    return {"config": asdict(config), "modules": modules}


def adapter_deltas(adapter: dict) -> dict[str, torch.Tensor]:
    """{"<module>.weight": (alpha / r) * B A} — keyed like the base model's
    state_dict, so it lines up with TaskVector's keys."""
    scaling = adapter["config"]["alpha"] / adapter["config"]["r"]
    return {
        f"{name}.weight": (ab["B"] @ ab["A"]) * scaling
        for name, ab in adapter["modules"].items()
    }


@torch.no_grad()
def merge_adapter_into(model: nn.Module, adapter: dict) -> None:
    """W += (alpha / r) * B A, in place, on a model without LoRALinear
    wrappers (O-LoRA eq. 9)."""
    for key, delta in adapter_deltas(adapter).items():
        param = model.get_parameter(key)
        param.add_(delta.to(device=param.device, dtype=param.dtype))


def adapters_to_vector(adapters: list[dict]) -> dict[str, torch.Tensor]:
    """The task vector W_t - W_0 of the model after the last of `adapters`,
    given all of them in training order."""
    vector: dict[str, torch.Tensor] = {}
    for adapter in adapters:
        for key, delta in adapter_deltas(adapter).items():
            vector[key] = vector[key] + delta if key in vector else delta
    return vector
