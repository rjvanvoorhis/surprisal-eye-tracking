import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
from collections import defaultdict

from common.utils import cache_result
from common.interfaces import SizedIterable
from common.constants import Constants


MODEL_NAME = Constants.LLAMA_MODEL_NAME  # Change to LLaMA model name

EOT_TOKEN = "<|endoftext|>"


@cache_result
def get_tokenizer() -> LlamaTokenizer:
    result = LlamaTokenizer.from_pretrained(MODEL_NAME)
    if not result.eos_token:
        result.eos_token = EOT_TOKEN
    return result


@cache_result
def get_model() -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)


def get_word_ids(tokens: SizedIterable[int], words: SizedIterable[str]):
    import json

    """
    Manually generate word_ids for a non-fast tokenizer.
    This function will map each token to the word it belongs to.
    """
    tokenizer = get_tokenizer()
    decoded = [tokenizer.decode(token).lower() for token in tokens][1:]
    word_ids = [0]
    for idx, word in enumerate(words, start=1):
        word = word.lower()
        while all([word, decoded]):
            next_token = decoded[0]
            if not word.startswith(next_token):
                break
            word = word.removeprefix(next_token)
            word_ids.append(idx)
            decoded = decoded[1:]
    return word_ids


def llama_surprisal(text: SizedIterable[str]):

    # Tokenizing the input text, similar to how GPT-2 was tokenized
    tokenized = get_tokenizer()(text, is_split_into_words=True, return_tensors="pt")
    input_ids = tokenized["input_ids"].squeeze(0)  # Flatten input_ids

    tokens = input_ids.tolist()  # Convert input_ids to a list of token ids
    words = text  # Original words in the input

    # Manually generate word_ids
    word_ids = get_word_ids(tokens, words)
    # print("Generated Word IDs:", word_ids)

    with torch.no_grad():
        # Forward pass through the model
        output = get_model()(
            input_ids.unsqueeze(0), attention_mask=tokenized["attention_mask"]
        )

    logits = output.logits.squeeze(0)  # Logits of shape (seq_len, vocab_size)
    target_ids = input_ids[1:]  # Skip the EOT token
    predictions = logits[:-1]  # Removing the last token's logits for prediction

    log_probabilities = F.log_softmax(predictions, dim=-1)  # Applying log softmax
    token_surprisals = -1 * log_probabilities[torch.arange(len(target_ids)), target_ids]

    # Aggregate surprisal by word
    word_surprisals = defaultdict(float)
    for idx, word_id in enumerate(word_ids[1:]):  # Skipping the first token (EOT)
        if word_id is not None:
            word_surprisals[word_id] += token_surprisals[idx].item()

    # Return the aggregated surprisal values for words
    return [word_surprisals[idx] for idx in sorted(word_surprisals)]


if __name__ == "__main__":
    print(llama_surprisal("I ride the bus every day to get to work".split()))
