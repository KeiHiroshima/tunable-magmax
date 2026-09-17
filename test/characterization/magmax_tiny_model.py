"""A few-hundred-parameter stand-in for ImageEncoder / BertClassifier.

Lives in its own importable module (not in conftest.py) because
src/utils.py::torch_save pickles the *whole model object*, so the class must
be re-importable by its qualified name when torch_load unpickles it.
"""

import torch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(5, 4)
        self.layer2 = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))
