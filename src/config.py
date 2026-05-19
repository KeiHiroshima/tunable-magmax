import os

BASE_DIR = os.environ.get("MAGMAX_BASE_DIR", "/mnt/ssd/KeiHiroshima/magmax_offline")
DATA_DIR = os.environ.get("MAGMAX_DATA_DIR", "/mnt/ssd/KeiHiroshima/data")
OPENCLIP_CACHE_DIR = os.path.join(BASE_DIR, "checkpoints/ViT-B-16/cachedir/open_clip")


def get_zeroshot_checkpoint(model_name):
    return os.path.join(BASE_DIR, "checkpoints", model_name, "zeroshot.pt")
