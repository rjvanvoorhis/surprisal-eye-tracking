import typing

import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertTokenizerFast

from common.utils import cache_result
from common.interfaces import SizedIterable
from common.constants import Constants

MODEL_NAME = Constants.BERT_MODEL_NAME


@cache_result
def get_tokenizer() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(MODEL_NAME)


@cache_result
def get_model() -> BertForMaskedLM:
    return BertForMaskedLM.from_pretrained(MODEL_NAME)


def bert_surprisal(text: SizedIterable[str]) -> SizedIterable[float]:
    tokenizer = get_tokenizer()
    tokenized = tokenizer(
        text,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    raw_inputs = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    word_ids = tokenized.word_ids()
    surprisals = []
    for word_id in sorted(set(filter(lambda x: x is not None, word_ids))):
        masked_input = raw_inputs.clone()
        to_mask = [idx for idx, value in enumerate(word_ids) if value == word_id]
        target_ids = raw_inputs[0, to_mask]
        masked_input[0, to_mask] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = get_model()(masked_input, attention_mask=attention_mask).logits
        log_probabilities = F.log_softmax(logits[0, to_mask], dim=-1)
        target_log_probabilities = log_probabilities[
            torch.arange(len(to_mask)), target_ids
        ]
        surprisals.append((-1 * target_log_probabilities.sum()).item())

    return surprisals


if __name__ == "__main__":
    print(bert_surprisal(["This is my sample sentence.".split()]))
