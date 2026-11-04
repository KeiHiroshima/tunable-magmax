import json
import logging
import os
from logging import config, getLogger

import numpy as np
import torch
import torchvision.transforms as T
import tqdm

from src.datasets.common import maybe_dictionarize


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


transform = T.Compose(
    [
        # T.ToPILImage(),
        T.Resize((224, 224)),
        T.RandomResizedCrop(
            size=(224, 224),
            scale=(0.9, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


def setup_logging(config_path="logging_config.json", level=None):
    """
    Args:
        config_path:
        level: ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    with open(os.path.join("configs", config_path), "r") as f:
        log_config = json.load(f)
    config.dictConfig(log_config)
    logger = getLogger("root")
    logger.info(f"Logging configured from {config_path}")

    if level:
        logger.setLevel(getattr(logging, level.upper()))
        logger.info(f"Log level set to {level.upper()}")

    return logger


@torch.no_grad()
def do_eval(model, dl, device, flag_data_parallel=False):
    correct, n = 0.0, 0.0
    device_to_pass = "cuda:0" if flag_data_parallel else device
    model.eval()
    for data in tqdm.tqdm(dl):
        data = maybe_dictionarize(data)
        try:
            x = data["images"].to(device_to_pass)
            y = data["labels"].to(device_to_pass)
        except Exception:
            x = data["image"].to(device_to_pass)
            y = data["label"].to(device_to_pass)

        logits = get_logits(x, model)
        if flag_data_parallel:
            pred = logits.argmax(dim=1, keepdim=True).to("cpu")
            y = y.to("cpu")
        else:
            pred = logits.argmax(dim=1, keepdim=True).to(device)
        correct += pred.eq(y.view_as(pred)).sum().item()
        n += y.size(0)

    metrics = {"top1": correct / n}

    # clean up GPU memory
    del x, y, logits, pred
    torch.cuda.empty_cache()

    # convert to float
    for key in metrics:
        metrics[key] = float(metrics[key])

    return metrics


def is_freezed_parameter(task_vectors):
    return all(torch.all(tv == 0) for tv in task_vectors)
