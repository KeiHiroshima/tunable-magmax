import torch
from transformers import AutoTokenizer, EncoderDecoderModel, PreTrainedModel, T5ForConditionalGeneration

# The CITB backend's model (build_bert2bert below). StdCL/LSB use T5 instead
# (build_t5), named by --model.
BASE_MODEL_NAME = "bert-base-uncased"


def build_t5(model_name: str) -> PreTrainedModel:
    """T5 for StdCL/LSB (O-LoRA's T5 setting): t5-base or t5-large, generating
    each task's label string.

    Loaded in fp32 explicitly — transformers 5 loads in the checkpoint's own
    dtype by default, and the V100s these runs target have no bf16 while T5
    is known to overflow in fp16. dropout_rate 0.1 is O-LoRA's value (also
    T5's own default, set here so it cannot drift with the checkpoint).
    """
    return T5ForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float32, dropout_rate=0.1
    )


def freeze_shared_embeddings(model: PreTrainedModel) -> None:
    """Keep T5's shared token embedding (tied to the encoder/decoder input
    embeddings and, in the original T5 checkpoints, to lm_head) out of
    training, as O-LoRA does for every T5 run.

    Besides following the reference setup, this keeps every tied alias of
    that one tensor at a zero task vector: merge methods treat state_dict keys
    independently, and a tied tensor listed under several keys could
    otherwise be merged to a different value under each.
    """
    for p in model.get_input_embeddings().parameters():
        p.requires_grad = False
    output = model.get_output_embeddings()
    if output is not None:
        for p in output.parameters():
            p.requires_grad = False


def build_bert2bert(model_name: str = BASE_MODEL_NAME) -> PreTrainedModel:
    """Encoder-decoder for CITB/SuperNI tasks (seq2seq generation).

    BERT is encoder-only, so generation needs an encoder-decoder wrapper.
    This warm-starts both towers from the same BERT-base checkpoint instead
    of introducing a T5 (or ViT) model, per the "BERT only, ViT-scale" design
    decision. Total size (~220M) is comparable to T5-base.
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
