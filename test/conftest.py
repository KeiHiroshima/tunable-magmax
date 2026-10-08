import os

# Keep HF downloads (models + datasets) inside the repo's own gitignored
# `datasets/` dir instead of the shared `~/.cache/huggingface`, matching the
# vision pipeline's MAGMAX_DATA_DIR convention.
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(__file__), "..", "datasets", "hf_cache")
)
