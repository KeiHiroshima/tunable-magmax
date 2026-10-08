"""--finetune_mode lora (src/nlp/lora.py and its use in
src/nlp/finetune_nlp.py / src/backends/nlp_classification_backend.py).

Only adapters are saved, and every model the merge step scores is rebuilt
from them, so the claims that matter are the reconstruction identities:

  * an adapter merged into the weights computes what the LoRA-wrapped model
    computed (O-LoRA eq. 9);
  * the task vector built from adapters 0..t is exactly W_t - W_0, i.e.
    applying it to the zero-shot model gives the model that task t+1 started
    from.

Everything runs on a tiny randomly initialised T5, offline.
"""

from argparse import Namespace

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from src.backends import nlp_classification_backend as backend
from src.nlp import finetune_nlp
from src.nlp.lora import (
    LoraConfig,
    LoRALinear,
    adapter_deltas,
    adapters_to_vector,
    extract_adapter,
    inject_lora,
    merge_adapter_into,
)
from src.paths import adapter_path, finetuned_path
from src.utils import torch_save


def _tiny_t5(seed=0):
    torch.manual_seed(seed)
    config = T5Config(
        vocab_size=64, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_decoder_layers=2,
        num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
    )
    return T5ForConditionalGeneration(config).eval()


def _inputs():
    g = torch.Generator().manual_seed(1)
    return dict(
        input_ids=torch.randint(2, 64, (3, 7), generator=g),
        attention_mask=torch.ones(3, 7, dtype=torch.long),
        labels=torch.randint(2, 64, (3, 4), generator=g),
    )


def _perturb_lora(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.normal_(generator=g)
                m.lora_B.normal_(std=0.05, generator=g)


def _logits(model):
    with torch.no_grad():
        return model.eval()(**_inputs()).logits


def test_lora_wraps_q_and_v_of_every_attention_block_and_trains_only_them():
    model = _tiny_t5()
    names = inject_lora(model, LoraConfig())

    # 2 encoder layers x self-attn + 2 decoder layers x (self + cross) = 6 blocks
    assert len(names) == 6 * 2
    assert {n.split(".")[-1] for n in names} == {"q", "v"}
    assert any("EncDecAttention" in n for n in names)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_A", "lora_B")) for n in trainable)


def test_fresh_adapter_leaves_the_model_unchanged():
    """B starts at zero, so injecting must not move the model — otherwise a
    task would start from somewhere other than the previous task's weights."""
    model = _tiny_t5()
    before = _logits(model)
    inject_lora(model, LoraConfig())

    assert torch.allclose(_logits(model), before)


def test_merged_adapter_computes_what_the_lora_model_computed():
    lora_model = _tiny_t5()
    inject_lora(lora_model, LoraConfig())
    _perturb_lora(lora_model, seed=3)
    expected = _logits(lora_model)

    merged = _tiny_t5()  # same seed: the same base weights
    merge_adapter_into(merged, extract_adapter(lora_model, LoraConfig()))

    assert torch.allclose(_logits(merged), expected, atol=1e-5)


def test_adapter_deltas_are_keyed_like_the_base_state_dict():
    model = _tiny_t5()
    inject_lora(model, LoraConfig())
    deltas = adapter_deltas(extract_adapter(model, LoraConfig()))

    base_keys = set(_tiny_t5().state_dict())
    assert set(deltas) <= base_keys
    assert all(k.endswith((".q.weight", ".v.weight")) for k in deltas)


def test_task_vector_is_the_running_sum_of_adapters():
    adapters = []
    for seed in (1, 2, 3):
        m = _tiny_t5()
        inject_lora(m, LoraConfig())
        _perturb_lora(m, seed)
        adapters.append(extract_adapter(m, LoraConfig()))

    vector = adapters_to_vector(adapters)
    key = next(iter(vector))
    expected = sum(adapter_deltas(a)[key] for a in adapters)

    assert torch.allclose(vector[key], expected)


# --- the fine-tuning loop and the merge step, end to end ---------------------


@pytest.fixture
def lora_run(tmp_path, monkeypatch):
    """Runs finetune_task_sequence under LoRA with a stand-in for training
    that perturbs the adapter, recording the model each task starts from."""
    zeroshot = tmp_path / "zeroshot.pt"
    ckpt_dir = tmp_path / "run"
    monkeypatch.setattr(finetune_nlp, "get_zeroshot_checkpoint", lambda name: str(zeroshot))
    starts = []

    def train_task(model, task, args):
        # the weights this task started from, with its fresh (zero) adapter
        starts.append(_logits(model))
        _perturb_lora(model, seed=len(starts))

    def run(n_tasks=3):
        args = Namespace(model="tiny", device="cpu", sequential_finetuning=True)
        tasks = [Namespace(name=f"t{i}", num_labels=2) for i in range(n_tasks)]
        finetune_nlp.finetune_task_sequence(
            args, tasks, str(ckpt_dir),
            build_base_model=lambda name: _tiny_t5(),
            train_task=train_task,
            lora=LoraConfig(),
        )
        return str(zeroshot), str(ckpt_dir), starts

    return run


def test_lora_loop_saves_adapters_only(lora_run):
    _, ckpt_dir, _ = lora_run(n_tasks=3)

    import os

    assert all(os.path.exists(adapter_path(ckpt_dir, i)) for i in range(3))
    assert not any(os.path.exists(finetuned_path(ckpt_dir, i)) for i in range(3))


def test_each_task_starts_from_the_previous_tasks_merged_model(lora_run):
    """I3 under LoRA: task i starts from zero-shot + adapters 0..i-1, which is
    also what the merge step's task vector i-1 must reconstruct."""
    zeroshot, ckpt_dir, starts = lora_run(n_tasks=3)
    args = Namespace(finetune_mode="lora", sequential_finetuning=True)
    task_vectors = backend._load_task_vectors(args, ckpt_dir, zeroshot, n_tasks=3)

    for i in (1, 2):
        rebuilt = backend._apply_task_vector(task_vectors[i - 1], zeroshot, 1.0, "cpu")
        assert torch.allclose(_logits(rebuilt), starts[i], atol=1e-5)

    assert torch.allclose(starts[0], _logits(_tiny_t5()), atol=1e-6)  # I4


def test_resumed_lora_run_continues_from_the_existing_adapters(lora_run):
    """I2/I3: a rerun trains nothing that exists, and the next new task still
    starts from every adapter so far."""
    zeroshot, ckpt_dir, starts = lora_run(n_tasks=2)
    starts.clear()

    lora_run(n_tasks=3)

    assert len(starts) == 1  # only task 2 trained
    args = Namespace(finetune_mode="lora", sequential_finetuning=True)
    task_vectors = backend._load_task_vectors(args, ckpt_dir, zeroshot, n_tasks=3)
    rebuilt = backend._apply_task_vector(task_vectors[1], zeroshot, 1.0, "cpu")
    assert torch.allclose(_logits(rebuilt), starts[0], atol=1e-5)


def test_full_mode_task_vectors_reconstruct_the_checkpoints(tmp_path):
    zeroshot = tmp_path / "zeroshot.pt"
    torch_save(_tiny_t5(), str(zeroshot))
    ckpt_dir = tmp_path / "run"
    for i in range(2):
        m = _tiny_t5()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * (i + 1))
        torch_save(m, finetuned_path(str(ckpt_dir), i))

    args = Namespace(finetune_mode="full", sequential_finetuning=True)
    task_vectors = backend._load_task_vectors(args, str(ckpt_dir), str(zeroshot), n_tasks=2)
    rebuilt = backend._apply_task_vector(task_vectors[1], str(zeroshot), 1.0, "cpu")

    finetuned = torch.load(finetuned_path(str(ckpt_dir), 1), weights_only=False)
    assert torch.allclose(_logits(rebuilt), _logits(finetuned), atol=1e-5)
