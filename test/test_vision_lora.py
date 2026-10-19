"""--finetune_mode lora for the vision (CLIP ViT) pipeline: src/lora.py,
shared with the NLP pipeline, applied after src/modeling.py's
unpack_visual_attention.

The claims that matter are the same reconstruction identities as
test/test_nlp_lora.py, plus the one the vision side adds:

  * unpacking the image tower's attention changes nothing it computes;
  * src/lora.py wraps the image tower's q/v and nothing else;
  * a merged adapter, and the task vector built from the adapters, give the
    model the LoRA-wrapped one computed.

Everything runs on a tiny randomly initialised open_clip VisionTransformer,
offline.
"""

from argparse import Namespace

import torch
from open_clip.transformer import VisionTransformer
from torch import nn

from src.backends import vision_backend
from src.lora import (
    LoraConfig,
    LoRALinear,
    adapter_task_vectors,
    extract_adapter,
    inject_lora,
    merge_adapter_into,
)
from src.modeling import UnpackedMultiheadAttention, unpack_visual_attention

LAYERS = 2


class _Clip(nn.Module):
    """The two towers ImageEncoder.model has: an image tower whose attention
    gets unpacked, and a text tower that keeps nn.MultiheadAttention."""

    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer(
            image_size=32, patch_size=16, width=32, layers=LAYERS, heads=4,
            mlp_ratio=2.0, output_dim=16,
        )
        self.transformer = nn.Module()
        self.transformer.attn = nn.MultiheadAttention(16, 2, batch_first=True)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Clip()

    def forward(self, images):
        return self.model.visual(images)


def _encoder(seed=0):
    torch.manual_seed(seed)
    return _Encoder().eval()


def _images(seed=1):
    return torch.randn(3, 3, 32, 32, generator=torch.Generator().manual_seed(seed))


def _randomise_adapters(model, seed):
    """Stand in for training: B starts at zero, which would make every
    identity below hold trivially."""
    gen = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_B.data = torch.randn(module.lora_B.shape, generator=gen) * 0.1


@torch.no_grad()
def test_unpacking_the_attention_changes_nothing_it_computes():
    packed = _encoder()
    unpacked = unpack_visual_attention(_encoder())

    assert all(
        isinstance(block.attn, UnpackedMultiheadAttention)
        for block in unpacked.model.visual.transformer.resblocks
    )
    torch.testing.assert_close(unpacked(_images()), packed(_images()))


def test_lora_wraps_only_the_image_towers_q_and_v():
    model = unpack_visual_attention(_encoder())

    names = inject_lora(model, LoraConfig())

    assert names == [
        f"model.visual.transformer.resblocks.{i}.attn.{p}"
        for i in range(LAYERS)
        for p in ("q", "v")
    ]
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {f"{n}.lora_{m}" for n in names for m in ("A", "B")}


@torch.no_grad()
def test_merged_adapters_and_task_vectors_rebuild_the_lora_model():
    config = LoraConfig()
    base = unpack_visual_attention(_encoder())
    base_state = {k: v.clone() for k, v in base.state_dict().items()}

    # Two sequential tasks, as vision_backend runs them: task 1 starts from
    # the base with task 0's adapter merged in.
    adapters, lora_model = [], None
    for task in range(2):
        model = unpack_visual_attention(_encoder())
        for adapter in adapters:
            merge_adapter_into(model, adapter)
        inject_lora(model, config)
        _randomise_adapters(model, seed=task)
        adapters.append(extract_adapter(model, config))
        lora_model = model.eval()

    merged = unpack_visual_attention(_encoder())
    for adapter in adapters:
        merge_adapter_into(merged, adapter)
    torch.testing.assert_close(merged(_images()), lora_model(_images()))

    # The last task's vector, applied to the base, is that same model.
    vector = adapter_task_vectors(adapters, sequential=True)[-1]
    assert set(vector) == {
        f"model.visual.transformer.resblocks.{i}.attn.{p}.weight"
        for i in range(LAYERS)
        for p in ("q", "v")
    }
    merged_state = merged.state_dict()
    for key, delta in vector.items():
        torch.testing.assert_close(base_state[key] + delta, merged_state[key])


def test_lora_and_full_runs_do_not_share_a_checkpoint_directory():
    args = Namespace(
        model="ViT-B-16", dataset="CIFAR100", n_splits=5, split_strategy="class",
        sequential_finetuning=True, taskseq_pattern="A", epochs=10, seed=3,
    )
    full = vision_backend._ckpt_dir(Namespace(**vars(args), finetune_mode="full"))
    lora = vision_backend._ckpt_dir(Namespace(**vars(args), finetune_mode="lora"))

    assert full.endswith("class_incremental/CIFAR100-5/ft-pattern_A-epochs-10-seed:3")
    assert lora.endswith("class_incremental/CIFAR100-5-lora/ft-pattern_A-epochs-10-seed:3")
