import open_clip
import torch
from tqdm import tqdm

from src.datasets.registry import get_dataset
from src.datasets.templates import get_templates
from src.modeling import ClassificationHead, ImageEncoder


def _compute_zeroshot_weights(model, classnames, template, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [t(classname) for t in template]
            texts = open_clip.tokenize(texts).to(device)  # tokenize
            embeddings = model.encode_text(texts)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= model.logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    return zeroshot_weights


def build_classification_head(
    model, dataset_name, data_location, device, classnames, args=None
):
    template = get_templates(dataset_name)

    if not classnames:
        classnames = get_dataset(
            dataset_name,
            None,
            location=data_location,
            batch_size=args.batch_size,
            args_=args,
        ).classnames

    model.eval()
    model.to(device)

    zeroshot_weights = _compute_zeroshot_weights(model, classnames, template, device)
    return ClassificationHead(normalize=True, weights=zeroshot_weights)


def build_subset_classification_head(
    model, dataset_name, classes, data_location, device, args=None
):
    template = get_templates(dataset_name)
    dataset = get_dataset(dataset_name, None, location=data_location, args_=args)
    classnames = [dataset.classnames[i] for i in classes]

    model.eval()
    model.to(device)

    print("Building SUBSET classification head.")
    zeroshot_weights = _compute_zeroshot_weights(model, classnames, template, device)
    return ClassificationHead(normalize=True, weights=zeroshot_weights)


def get_classification_head(args, dataset_name, image_encoder=None, classnames=None):
    if isinstance(image_encoder, ImageEncoder) and image_encoder.has_lang():
        print("Using passed model to create classifier!")
        model = image_encoder.model
    else:
        model = ImageEncoder(args, keep_lang=True).model

    classification_head = build_classification_head(
        model, dataset_name, args.data_location, args.device, classnames, args=args
    )

    return classification_head
