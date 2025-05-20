import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from collections import defaultdict

from common.utils import cache_result
from common.interfaces import SizedIterable
from common.constants import Constants

MODEL_NAME = Constants.GPT_MODEL_NAME


EOT_TOKEN = "<|endoftext|>"


@cache_result
def get_tokenizer() -> GPT2TokenizerFast:
    return GPT2TokenizerFast.from_pretrained(MODEL_NAME, add_prefix_space=True)


@cache_result
def get_model() -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(MODEL_NAME)


def gpt_surprisal(text: SizedIterable[str]) -> SizedIterable[float]:
    tokenizer = get_tokenizer()
    model = get_model()
    tokenized = tokenizer(
        [EOT_TOKEN, *text],
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    word_ids = tokenized.word_ids()
    with torch.no_grad():
        output = model(
            tokenized["input_ids"], attention_mask=tokenized["attention_mask"]
        )

    logits = output.logits.squeeze(0)
    target_ids = tokenized["input_ids"].squeeze(0)[1:]
    predictions = logits[:-1]

    log_probabilities = F.log_softmax(predictions, dim=-1)
    token_surprisals = -1 * log_probabilities[torch.arange(len(target_ids)), target_ids]

    word_surprisals = defaultdict(float)
    for idx, word_id in enumerate(word_ids[1:]):
        if word_id is not None:
            word_surprisals[word_id] += token_surprisals[idx].item()

    return [word_surprisals[idx] for idx in sorted(word_surprisals)]


if __name__ == "__main__":
    print(gpt_surprisal("I ride the bus every day to get to work".split()))
