"""Maps --dataset to the backend module implementing it. Adding a new
modality/task family means adding one module under src/backends/ and one
entry here — no changes to any existing backend.
"""

from src.backends import nlp_classification_backend, nlp_seq2seq_backend, vision_backend

DATASET_TO_BACKEND = {
    "CIFAR100": vision_backend,
    "ImageNetR": vision_backend,
    "LSB": nlp_classification_backend,
    "CITB19": nlp_seq2seq_backend,
    "CITB38": nlp_seq2seq_backend,
}


def resolve_backend(dataset_name: str):
    try:
        return DATASET_TO_BACKEND[dataset_name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Registered: {sorted(DATASET_TO_BACKEND)}"
        ) from None
