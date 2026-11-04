import os

BASE_DIR = os.environ.get("MAGMAX_BASE_DIR", "/mnt/ssd/KeiHiroshima/tunable_magmax")
DATA_DIR = os.environ.get("MAGMAX_DATA_DIR", "/mnt/ssd/KeiHiroshima/data")
OPENCLIP_CACHE_DIR = os.path.join(BASE_DIR, "checkpoints/ViT-B-16/cachedir/open_clip")


def get_zeroshot_checkpoint(model_name):
    return os.path.join(BASE_DIR, "checkpoints", model_name, "zeroshot.pt")


def get_unpacked_zeroshot_checkpoint(model_name):
    """zeroshot.pt with its image-tower attention split into q/k/v layers
    (src/modeling.py's unpack_visual_attention): the same weights and outputs,
    but keyed so the vision LoRA task vectors apply to it."""
    return os.path.join(BASE_DIR, "checkpoints", model_name, "zeroshot_unpacked.pt")
