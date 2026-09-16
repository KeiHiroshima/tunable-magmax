import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, EncoderDecoderModel, PreTrainedModel

# bert-base-uncased: 12 layers, hidden 768, 12 heads, ~110M params — the same
# "Base" regime as the ViT-B/16 (~86M) used for the vision experiments.
BASE_MODEL_NAME = "bert-base-uncased"


class BertClassifier(nn.Module):
    """Encoder for Long Sequence Benchmark tasks (classification).

    Mirrors `src/modeling.py`'s ImageEncoder+head split for the vision
    pipeline: only `head` is swapped per task, so a task vector can be taken
    over every other parameter. Unlike the vision pipeline's CLIP zero-shot
    head (`src/heads.py`), BERT has no shared image-text embedding space to
    build a head from, so the head is just a randomly initialized nn.Linear.
    """

    def __init__(self, model_name: str = BASE_MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.head: nn.Linear | None = None

    def reset_head(self, num_labels: int) -> None:
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        pooled = self.encoder(input_ids, attention_mask=attention_mask).pooler_output
        return self.head(pooled)


def build_bert2bert(model_name: str = BASE_MODEL_NAME) -> PreTrainedModel:
    """Encoder-decoder for CITB/SuperNI tasks (seq2seq generation).

    BERT is encoder-only, so generation needs an encoder-decoder wrapper.
    This warm-starts both towers from the same BERT-base checkpoint instead
    of introducing a T5 (or ViT) model, per the "BERT only, ViT-scale" design
    decision. Total size (~220M) is roughly double a single BertClassifier,
    comparable to T5-base.
    """
    # decoder_start_token_id/pad_token_id/eos_token_id come from the
    # tokenizer, not BertConfig — BertConfig has no *_token_id fields.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    return model
