"""--finetune_mode lora for the vision (CLIP ViT) pipeline: src/lora.py,
shared with the NLP pipeline, applied after src/modeling.py's
unpack_visual_attention.

The claims that matter are the same reconstruction identities as
test/test_nlp_lora.py, plus the one the vision side adds:

  * unpacking the image tower's attention changes nothing it computes;
  * src/lora.py wraps the image tower's q/v and nothing else;
  * a merged adapter, and the task vector built from the adapters, give the
    model the LoRA-wrapped one computed;
  * under --lora_chain reuse, a split starts on the unmerged base with the
    previous split's adapter, and its task vector alone rebuilds its model.

Everything runs on a tiny randomly initialised open_clip VisionTransformer,
offline.
"""

from argparse import Namespace

import pytest
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


# --- --lora_chain reuse --------------------------------------------------------


@pytest.fixture
def saved_adapters(tmp_path):
    """The unpacked base on disk, and two saved adapters for it."""
    config = LoraConfig()
    base_path = tmp_path / "unpacked_zeroshot.pt"
    torch.save(unpack_visual_attention(_encoder()), base_path)
    paths = []
    for task in range(2):
        model = unpack_visual_attention(_encoder())
        inject_lora(model, config)
        _randomise_adapters(model, seed=task)
        paths.append(str(tmp_path / f"adapter_{task}.pt"))
        torch.save(extract_adapter(model, config), paths[-1])
    return str(base_path), paths


def _args(chain):
    return Namespace(finetune_mode="lora", sequential_finetuning=True, lora_chain=chain)


def _visual_state(model):
    return {k: v for k, v in model.state_dict().items() if k.startswith("model.visual.")}


def test_reuse_split_starts_on_the_unmerged_base(saved_adapters):
    base_path, paths = saved_adapters
    base = torch.load(base_path, weights_only=False)

    start = vision_backend._lora_start_encoder(_args("reuse"), base_path, paths)

    for key, value in _visual_state(base).items():
        assert torch.equal(start.state_dict()[key], value)


def test_merge_split_still_starts_with_the_earlier_adapters_merged_in(saved_adapters):
    base_path, paths = saved_adapters
    expected = torch.load(base_path, weights_only=False)
    for path in paths:
        merge_adapter_into(expected, torch.load(path, weights_only=False))

    start = vision_backend._lora_start_encoder(_args("merge"), base_path, paths)

    for key, value in _visual_state(expected).items():
        assert torch.equal(start.state_dict()[key], value)


def test_reuse_resumes_the_previous_splits_adapter(saved_adapters):
    _, paths = saved_adapters
    model = unpack_visual_attention(_encoder())
    inject_lora(model, LoraConfig())

    vision_backend._resume_adapter(model, paths, LoraConfig(), _args("reuse"))

    prev = torch.load(paths[-1], weights_only=False)["modules"]
    for name, m in model.named_modules():
        if isinstance(m, LoRALinear):
            assert torch.equal(m.lora_A, prev[name]["A"])
            assert torch.equal(m.lora_B, prev[name]["B"])


@pytest.mark.parametrize(
    "chain,adapters_so_far", [("reuse", "none"), ("merge", "all")],
    ids=["first-split", "merge-chain"],
)
def test_resume_leaves_a_fresh_adapter_otherwise(saved_adapters, chain, adapters_so_far):
    _, paths = saved_adapters
    model = unpack_visual_attention(_encoder())
    inject_lora(model, LoraConfig())

    vision_backend._resume_adapter(
        model, [] if adapters_so_far == "none" else paths, LoraConfig(), _args(chain)
    )

    for m in model.modules():
        if isinstance(m, LoRALinear):
            assert torch.count_nonzero(m.lora_B) == 0


@torch.no_grad()
def test_reuse_task_vector_alone_rebuilds_the_lora_model():
    """Two splits as vision_backend runs them under reuse: split 1 continues
    split 0's adapter on the unmerged base, and W_0 + tau_1 is that model."""
    config, args = LoraConfig(), _args("reuse")
    base = unpack_visual_attention(_encoder())
    base_state = {k: v.clone() for k, v in base.state_dict().items()}

    adapters, lora_model = [], None
    for task in range(2):
        model = unpack_visual_attention(_encoder())
        inject_lora(model, config)
        if adapters:
            vision_backend.load_adapter_into(model, adapters[-1], config)
        _randomise_adapters(model, seed=task)
        adapters.append(extract_adapter(model, config))
        lora_model = model.eval()

    vector = adapter_task_vectors(adapters, sequential=True, chain="reuse")[-1]
    rebuilt = unpack_visual_attention(_encoder())
    state = rebuilt.state_dict()
    for key, delta in vector.items():
        state[key] = base_state[key] + delta
    rebuilt.load_state_dict(state)

    torch.testing.assert_close(rebuilt.eval()(_images()), lora_model(_images()))


def test_reuse_and_merge_runs_do_not_share_a_checkpoint_directory():
    args = dict(
        model="ViT-B-16", dataset="CIFAR100", n_splits=5, split_strategy="class",
        sequential_finetuning=True, taskseq_pattern="A", epochs=10, seed=3,
        finetune_mode="lora",
    )
    merge = vision_backend._ckpt_dir(Namespace(**args, lora_chain="merge"))
    reuse = vision_backend._ckpt_dir(Namespace(**args, lora_chain="reuse"))

    assert merge.endswith("class_incremental/CIFAR100-5-lora/ft-pattern_A-epochs-10-seed:3")
    assert reuse.endswith("class_incremental/CIFAR100-5-lora-reuse/ft-pattern_A-epochs-10-seed:3")
