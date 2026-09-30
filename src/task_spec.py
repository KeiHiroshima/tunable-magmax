from dataclasses import dataclass
from typing import Literal, Optional

from torch.utils.data import DataLoader


@dataclass
class TaskSpec:
    """Benchmark-agnostic unit handed to the CL training loop.

    Both NLP benchmark families are seq2seq: StdCL/LSB (O-LoRA's T5 setting)
    generate a label string, CITB/SuperNI a free-form response. `task_type`
    is kept so a loop can still dispatch on it without either benchmark module
    depending on the other.
    """

    name: str
    train_loader: DataLoader
    eval_loader: DataLoader
    task_type: Literal["classification", "seq2seq"]
    num_labels: Optional[int] = None  # classification only
    # StdCL/LSB: the label strings the model is asked to generate, in the
    # order the prompt's "Option:" line lists them.
    labels: Optional[list[str]] = None
