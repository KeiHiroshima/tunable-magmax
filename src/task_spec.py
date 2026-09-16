from dataclasses import dataclass
from typing import Literal, Optional

from torch.utils.data import DataLoader


@dataclass
class TaskSpec:
    """Benchmark-agnostic unit handed to the CL training loop.

    LSB tasks are classification (a fresh linear head per task); CITB/SuperNI
    tasks are seq2seq (no head to swap, decoder produces the vocabulary
    directly). `task_type` lets the training loop dispatch between the two
    without either benchmark module depending on the other.
    """

    name: str
    train_loader: DataLoader
    eval_loader: DataLoader
    task_type: Literal["classification", "seq2seq"]
    num_labels: Optional[int] = None  # classification only
