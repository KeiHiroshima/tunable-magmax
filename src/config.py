import os

BASE_DIR = os.environ.get("MAGMAX_BASE_DIR", "YOUR_BASE_DIR_FOR_CHECKPOINTS")
DATA_DIR = os.environ.get("MAGMAX_DATA_DIR", "YOUR_DATA_DIR")
OPENCLIP_CACHE_DIR = os.path.join(BASE_DIR, "checkpoints/ViT-B-16/cachedir/open_clip")


def get_zeroshot_checkpoint(model_name):
    return os.path.join(BASE_DIR, "checkpoints", model_name, "zeroshot.pt")
