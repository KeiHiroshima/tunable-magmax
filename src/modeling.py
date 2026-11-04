from logging import getLogger

import open_clip
import torch

from . import utils

logger = getLogger(__name__)


class ImageEncoder(torch.nn.Module):
    def __init__(self, args, keep_lang=False):
        super().__init__()

        logger.info(f"Loading {args.model} pre-trained weights.")
        if "__pretrained__" in args.model:
            name, pretrained = args.model.split("__pretrained__")
        else:
            name = args.model
            pretrained = "openai"
        logger.info(
            f"Loading pre-trained weights for model {name}, pretrained ckpt: {pretrained}."
        )
        self.model, self.train_preprocess, self.val_preprocess = (
            open_clip.create_model_and_transforms(
                name, pretrained=pretrained, cache_dir=args.openclip_cachedir
            )
        )

        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        utils.torch_save(self, filename)

    def has_lang(self) -> bool:
        return hasattr(self.model, "transformer")

    def freeze_lang(self):
        for pg in self.model.transformer.parameters():
            pg.requires_grad = False

    @classmethod
    def load(cls, filename):
        print(f"Loading image encoder from {filename}")
        return torch.load(filename, weights_only=False)


class UnpackedMultiheadAttention(torch.nn.Module):
    """nn.MultiheadAttention with its packed in_proj split into separate q, k
    and v nn.Linear layers. Computes the same function as the module it was
    built from.

    open_clip's ViT keeps q, k and v in one (3E, E) in_proj_weight, while
    src/lora.py — written for T5, where they are separate Linear layers named
    q/k/v — wraps nn.Linear modules by name. Splitting them lets the vision
    pipeline use src/lora.py unchanged, and gives the task vector separate
    q/v keys as under T5: a LoRA delta on the packed weight would carry an
    all-zero k block that the masked merge would spend its budget on.
    """

    def __init__(self, mha: torch.nn.MultiheadAttention):
        super().__init__()
        assert mha._qkv_same_embed_dim and mha.in_proj_bias is not None
        assert mha.bias_k is None and not mha.add_zero_attn
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = mha.head_dim
        self.dropout = mha.dropout
        self.batch_first = mha.batch_first

        weights = mha.in_proj_weight.detach().chunk(3)
        biases = mha.in_proj_bias.detach().chunk(3)
        for name, w, b in zip(("q", "k", "v"), weights, biases):
            linear = torch.nn.Linear(self.embed_dim, self.embed_dim)
            linear.weight = torch.nn.Parameter(w.clone())
            linear.bias = torch.nn.Parameter(b.clone())
            setattr(self, name, linear)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj.weight = torch.nn.Parameter(mha.out_proj.weight.detach().clone())
        self.out_proj.bias = torch.nn.Parameter(mha.out_proj.bias.detach().clone())

    def _heads(self, x):
        n, length, _ = x.shape
        return x.view(n, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        # Only the call open_clip's ResidualAttentionBlock makes is supported.
        assert not need_weights
        if not self.batch_first:
            query, key, value = (x.transpose(0, 1) for x in (query, key, value))
        n, length, _ = query.shape
        out = torch.nn.functional.scaled_dot_product_attention(
            self._heads(self.q(query)),
            self._heads(self.k(key)),
            self._heads(self.v(value)),
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = self.out_proj(out.transpose(1, 2).reshape(n, length, self.embed_dim))
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out, None


def unpack_visual_attention(image_encoder: "ImageEncoder") -> "ImageEncoder":
    """Replace every attention block of the image tower, in place, with an
    UnpackedMultiheadAttention. The text tower is left as it is, so
    src/lora.py's inject_lora only ever matches the image tower's q/v."""
    for block in image_encoder.model.visual.transformer.resblocks:
        block.attn = UnpackedMultiheadAttention(block.attn)
    return image_encoder


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading classification head from {filename}")
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def freeze_lang(self):
        self.image_encoder.freeze_lang()

    def freeze(self):
        self.freeze_head()
        for pg in self.image_encoder.parameters():
            pg.requires_grad = False

    def forward(self, inputs, return_features=False):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs if not return_features else (outputs, features)

    def __call__(self, inputs, return_features=False):
        return self.forward(inputs, return_features)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading image classifier from {filename}")
        return utils.torch_load(filename)
