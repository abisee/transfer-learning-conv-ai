"""
A script (based on interact.py) to generate a batch of responses to a history.
"""


import logging
import random

import torch
import time

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import add_special_tokens_
from interact import sample_sequence

def get_response(history, model, model_checkpoint, max_history=2, device="cuda" if torch.cuda.is_available() else "cpu",
                 no_sample=False, max_length=20, min_length=1, seed=42, temperature=0.7, top_k=0, top_p=0.9,
                 num_samples=1):
    """
    Inputs:
        history: list (odd length >=1) of list of strings. first string is user.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info('Starting get_response')

    logger.info("device: ", device)

    # Check history is odd length >=1
    assert len(history) >= 1
    assert len(history) % 2 == 1

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" in model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" in model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(model_checkpoint)
    model.to(device)
    add_special_tokens_(model, tokenizer)

    logger.info('top_k: {}, top_p: {}, temperature: {}, num_samples: {}'.format(
        top_k, top_p, temperature, num_samples))

    # Get history_tokenized, which should be a list of list of ints. each list represents a turn.
    history = history[-(2 * max_history + 1):]
    logger.info(f'Using this history: {history}')
    history_tokenized = [tokenizer.encode(utterance) for utterance in history]

    # Get responses
    t0 = time.time()
    with torch.no_grad():
        # out_ids is a list length num_samples of lists of ints
        logger.info(f'Batch-sampling {num_samples} responses')
        out_ids = sample_sequence(history_tokenized, tokenizer, model, device=device, no_sample=no_sample,
                                  max_length=max_length, min_length=min_length, temperature=temperature,
                                  top_k=top_k, top_p=top_p, num_samples=num_samples, current_output=None)
        logger.info(f'Sampling {num_samples} samples (max length {max([len(sample) for sample in out_ids])}) took {time.time() - t0} seconds')
    responses = [tokenizer.decode(out, skip_special_tokens=True) for out in out_ids]
    logger.info(f'Responses: {responses}')
    return responses


if __name__ == "__main__":
    history = ["i'm having such a bad day i'm feeling sick"]
    get_response(history, model='gpt2-medium', model_checkpoint='runs/Jan04_22-40-10_ip-172-31-71-210_gpt2-medium',
                 num_samples=10, top_p=0, top_k=5)


