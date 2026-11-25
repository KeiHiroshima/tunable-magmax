"""--finetune_mode lora (src/lora.py and its use in
src/nlp/finetune_nlp.py / src/backends/nlp_classification_backend.py).

Only adapters are saved, and every model the merge step scores is rebuilt
from them, so the claims that matter are the reconstruction identities:

  * an adapter merged into the weights computes what the LoRA-wrapped model
    computed (O-LoRA eq. 9);
  * the task vector built from adapters 0..t is exactly W_t - W_0, i.e.
    applying it to the zero-shot model gives the model that task t+1 started
    from;
  * under --lora_chain reuse, every task starts on the unmerged W_0 with the
    previous task's A and B, and task t's vector is adapter t's delta alone.

Everything runs on a tiny randomly initialised T5, offline.
"""

from argparse import Namespace

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from src.backends import nlp_classification_backend as backend
from src.nlp import finetune_nlp
from src.lora import (
    LoraConfig,
    LoRALinear,
    adapter_deltas,
    adapters_to_vector,
    adapter_task_vectors,
    extract_adapter,
    inject_lora,
    load_adapter_into,
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


def test_core_space_from_saved_adapters_covers_the_task_vectors(lora_run):
    """--merge_space core on a real LoRA run: the bases built from the saved
    adapters are keyed like the task vectors, the round trip is lossless,
    and the linear "average" merge yields the same model in either space."""
    from src.merging.registry import apply_merge, get_merge_spec

    zeroshot, ckpt_dir, _ = lora_run(n_tasks=3)
    args = Namespace(finetune_mode="lora", sequential_finetuning=True, merge_space="core")
    task_vectors = backend._load_task_vectors(args, ckpt_dir, zeroshot, n_tasks=3)
    core_space = backend._load_core_space(args, ckpt_dir, n_tasks=3)

    assert set(core_space.bases) == set(task_vectors[0].vector)
    for tv, core in zip(task_vectors, core_space.to_core(task_vectors)):
        back = core_space.from_core(core)
        for key, tau in tv.vector.items():
            assert torch.allclose(back.vector[key], tau, atol=1e-6)

    spec = get_merge_spec("average")
    full = backend._apply_task_vector(apply_merge(spec, task_vectors), zeroshot, 0.5, "cpu")
    core = backend._apply_task_vector(
        apply_merge(spec, task_vectors, core_space=core_space), zeroshot, 0.5, "cpu"
    )
    assert torch.allclose(_logits(core), _logits(full), atol=1e-5)


def test_full_merge_space_builds_no_core_space(lora_run):
    _, ckpt_dir, _ = lora_run(n_tasks=2)
    args = Namespace(finetune_mode="lora", merge_space="full")
    assert backend._load_core_space(args, ckpt_dir, n_tasks=2) is None


# --- --lora_chain reuse --------------------------------------------------------


def _lora_modules(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


@pytest.fixture
def reuse_run(tmp_path, monkeypatch):
    """Runs finetune_task_sequence under --lora_chain reuse with a stand-in
    for training that perturbs the adapter, recording for every task the
    state it started from: output, frozen base weights, and A/B."""
    zeroshot = tmp_path / "zeroshot.pt"
    ckpt_dir = tmp_path / "run"
    monkeypatch.setattr(finetune_nlp, "get_zeroshot_checkpoint", lambda name: str(zeroshot))
    starts = []

    def train_task(model, task, args):
        modules = _lora_modules(model)
        starts.append(dict(
            logits=_logits(model),
            base={n: m.base.weight.detach().clone() for n, m in modules.items()},
            A={n: m.lora_A.detach().clone() for n, m in modules.items()},
            B={n: m.lora_B.detach().clone() for n, m in modules.items()},
        ))
        _perturb_lora(model, seed=len(starts))

    def run(n_tasks=3):
        args = Namespace(
            model="tiny", device="cpu", sequential_finetuning=True, lora_chain="reuse"
        )
        tasks = [Namespace(name=f"t{i}", num_labels=2) for i in range(n_tasks)]
        finetune_nlp.finetune_task_sequence(
            args, tasks, str(ckpt_dir),
            build_base_model=lambda name: _tiny_t5(),
            train_task=train_task,
            lora=LoraConfig(),
        )
        return str(zeroshot), str(ckpt_dir), starts

    return run


def _reuse_args():
    return Namespace(finetune_mode="lora", sequential_finetuning=True, lora_chain="reuse")


def _saved(ckpt_dir, i):
    return torch.load(adapter_path(ckpt_dir, i), weights_only=False)


def test_reuse_never_merges_an_adapter_into_the_base(reuse_run):
    zeroshot, _, starts = reuse_run(n_tasks=3)
    w0 = torch.load(zeroshot, weights_only=False).state_dict()

    for start in starts:
        for name, weight in start["base"].items():
            assert torch.equal(weight, w0[f"{name}.weight"])


def test_reuse_task_starts_from_the_previous_tasks_adapter(reuse_run):
    _, ckpt_dir, starts = reuse_run(n_tasks=3)

    assert all(torch.count_nonzero(b) == 0 for b in starts[0]["B"].values())  # fresh
    for t in (1, 2):
        prev = _saved(ckpt_dir, t - 1)["modules"]
        for name in starts[t]["A"]:
            assert torch.equal(starts[t]["A"][name], prev[name]["A"])
            assert torch.equal(starts[t]["B"][name], prev[name]["B"])


def test_reuse_task_starts_from_the_previous_tasks_model(reuse_run):
    """I3 under reuse: task t starts from what task t-1's vector rebuilds."""
    zeroshot, ckpt_dir, starts = reuse_run(n_tasks=3)
    task_vectors = backend._load_task_vectors(_reuse_args(), ckpt_dir, zeroshot, n_tasks=3)

    for t in (1, 2):
        rebuilt = backend._apply_task_vector(task_vectors[t - 1], zeroshot, 1.0, "cpu")
        assert torch.allclose(_logits(rebuilt), starts[t]["logits"], atol=1e-5)
    assert torch.allclose(starts[0]["logits"], _logits(_tiny_t5()), atol=1e-6)  # I4


def test_reuse_task_vector_is_one_adapters_delta(reuse_run):
    zeroshot, ckpt_dir, _ = reuse_run(n_tasks=3)
    task_vectors = backend._load_task_vectors(_reuse_args(), ckpt_dir, zeroshot, n_tasks=3)
    r = LoraConfig().r

    for t, tv in enumerate(task_vectors):
        expected = adapter_deltas(_saved(ckpt_dir, t))
        assert tv.vector.keys() == expected.keys()
        for key, tau in tv.vector.items():
            assert torch.equal(tau, expected[key])
            assert torch.linalg.matrix_rank(tau) <= r


def test_resumed_reuse_run_continues_from_the_last_adapter(reuse_run):
    """I2/I3: a rerun trains only the missing task, from the last saved adapter."""
    _, ckpt_dir, starts = reuse_run(n_tasks=2)
    starts.clear()

    reuse_run(n_tasks=3)

    assert len(starts) == 1  # only task 2 trained
    prev = _saved(ckpt_dir, 1)["modules"]
    for name in starts[0]["A"]:
        assert torch.equal(starts[0]["A"][name], prev[name]["A"])
        assert torch.equal(starts[0]["B"][name], prev[name]["B"])


def test_reuse_and_merge_runs_do_not_share_a_checkpoint_directory():
    args = dict(
        model="t5-base", dataset="LSB", taskseq_pattern="4", epochs=1, seed=3,
        sequential_finetuning=True, finetune_mode="lora",
    )
    merge = backend._ckpt_dir(Namespace(**args, lora_chain="merge"))
    reuse = backend._ckpt_dir(Namespace(**args, lora_chain="reuse"))
    unset = backend._ckpt_dir(Namespace(**args))

    assert merge.endswith("nlp_classification/LSB-lora/ft-pattern_4-epochs-1-seed:3")
    assert reuse.endswith("nlp_classification/LSB-lora-reuse/ft-pattern_4-epochs-1-seed:3")
    assert unset == merge


# --- load_adapter_into / adapter_task_vectors ----------------------------------


def _adapter_from(config=LoraConfig(), seed=1):
    model = _tiny_t5()
    inject_lora(model, config)
    _perturb_lora(model, seed=seed)
    return extract_adapter(model, config)


def test_load_adapter_into_copies_a_and_b():
    adapter = _adapter_from()
    model = _tiny_t5()
    inject_lora(model, LoraConfig())

    load_adapter_into(model, adapter, LoraConfig())

    for name, m in _lora_modules(model).items():
        assert torch.equal(m.lora_A, adapter["modules"][name]["A"])
        assert torch.equal(m.lora_B, adapter["modules"][name]["B"])


def _missing_module(adapter):
    adapter["modules"].pop(next(iter(adapter["modules"])))


def _wrong_shape(adapter):
    ab = next(iter(adapter["modules"].values()))
    ab["A"] = ab["A"][:-1]


def _other_alpha(adapter):
    adapter["config"]["alpha"] = 16  # same r, so every shape still matches


@pytest.mark.parametrize(
    "corrupt", [_other_alpha, _missing_module, _wrong_shape],
    ids=["alpha", "missing-module", "shape"],
)
def test_load_adapter_into_rejects_a_mismatched_adapter(corrupt):
    adapter = _adapter_from()
    corrupt(adapter)
    model = _tiny_t5()
    inject_lora(model, LoraConfig())
    before = {n: m.lora_A.detach().clone() for n, m in _lora_modules(model).items()}

    with pytest.raises(ValueError):
        load_adapter_into(model, adapter, LoraConfig())
    # nothing was copied before the check failed
    for name, m in _lora_modules(model).items():
        assert torch.equal(m.lora_A, before[name])


def test_reuse_task_vectors_are_each_adapter_alone():
    adapters = [_adapter_from(seed=s) for s in (1, 2, 3)]

    reuse = adapter_task_vectors(adapters, sequential=True, chain="reuse")
    merge = adapter_task_vectors(adapters, sequential=True)  # default chain

    for t, adapter in enumerate(adapters):
        key = next(iter(reuse[t]))
        assert torch.equal(reuse[t][key], adapter_deltas(adapter)[key])
        assert torch.allclose(merge[t][key], adapters_to_vector(adapters[: t + 1])[key])
