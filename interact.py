# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from .train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from .utils import get_dataset_personalities, download_pretrained_model


@dataclass
class DecodeConfig:
    # Default is the recommended settings in transfer-learning-conv-ai
    no_sample: bool = False
    min_length: int = 1
    max_length: int = 20
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 0.9
    max_history: int = 2
    num_samples: int = 10


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (batch_size, vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # sorted_indices_to_remove is shape (batch_size, vocab_size) containing bools corresponding to sorted_indices.
        # Each row has Falses, then Trues.
        # For each row, get the index (in sorted_indices_to_remove) of the last False
        num_falses = sorted_indices_to_remove.size(1) - sorted_indices_to_remove.sum(dim=1)  # num false per row
        last_false = num_falses - 1  # idx of last false per row. shape (batch_size)

        # For each row, get the vocab-index of the last "False" token (i.e. least prob token that won't be masked)
        least_prob_index = sorted_indices[range(sorted_indices.size(0)), last_false]  # shape (batch_size)

        # For each row, get the logit for the least probable unmasked token
        cutoff_logits = logits[range(sorted_indices.size(0)), least_prob_index]  # shape (batch_size)

        # For each row, set everything lower than cutoff_logits to filter_value
        indices_to_remove = logits < cutoff_logits.unsqueeze(1)
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(history, tokenizer, model, device="cuda" if torch.cuda.is_available() else "cpu",
                    no_sample=False, max_length=20, min_length=1, temperature=0.7, top_k=0, top_p=0.9,
                    num_samples=1, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []  # now current_output is a list of ints

    # Initialize current_output for each sample
    current_outputs = [[i for i in current_output] for _ in range(num_samples)]

    # Record whether each sample is finished or not
    finished = [False for _ in range(num_samples)]

    for i in range(max_length):
        # print(f'step {i} of max {max_length}')
        instances = [build_input_from_segments(history, current_output, tokenizer, with_eos=False) for current_output in current_outputs]

        # input_ids and token_type_ids are both tensors shape (num_samples, seqlen)
        input_ids = torch.tensor([instance["input_ids"] for instance in instances], device=device)
        token_type_ids = torch.tensor([instance["token_type_ids"] for instance in instances], device=device)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]

        # now logits is shape (num_samples, seqlen, vocab_size)
        logits = logits[:, -1, :].squeeze(1) / temperature  # take predictions for last timestep. shape (num_samples, vocab_size)
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        prev = torch.topk(probs, 1)[1] if no_sample else torch.multinomial(probs, 1)

        # If any sampled tokens (in prev) are in special_tokens_ids, but we haven't reached min_length yet, resample
        for idx, p in enumerate(prev):
            if i < min_length and p.item() in special_tokens_ids:
                while p.item() in special_tokens_ids:
                    if probs[idx].max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev[idx] = torch.multinomial(probs[idx], num_samples=1)

        # Update which samples have finished
        finished = [f or p.item() in special_tokens_ids for (f,p) in zip(finished, prev)]

        # If they've all finished, quit
        if all(finished):
            break

        # Otherwise append the latest tokens and continue
        for current_output, p in zip(current_outputs, prev):
            current_output.append(p.item())

    # Within each sample, remove everything after the first special token
    for sample_num, current_output in enumerate(current_outputs):
        first_special_token_idx = next((idx for idx, val in enumerate(current_output) if val in special_tokens_ids), -1)  # will be -1 if there is no special token
        if first_special_token_idx != -1:
            current_outputs[sample_num] = current_output[:first_special_token_idx]

    # Separate out the unfinished ones
    finished_outputs = [output for (output, is_finished) in zip(current_outputs, finished) if is_finished]
    unfinished_outputs = [output for (output, is_finished) in zip(current_outputs, finished) if not is_finished]

    return finished_outputs, unfinished_outputs

def load_model(model, model_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Getting pretrained model and tokenizer (device={device})...")
    tokenizer_class = GPT2Tokenizer if "gpt2" in model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" in model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(model_checkpoint)
    model.to(device)
    add_special_tokens_(model, tokenizer)
    return model, tokenizer


def batch_decode(model, tokenizer, history, config):
    """
    Generate num_samples responses to history.

    @param model:
    @param tokenizer:
    @param history: list of strings (each one an utterance)
    @param config:
    @return: finished_responses: list (length <=config.num_samples) of strings, each a response to history.
    """
    # Get history_tokenized, which should be a list of list of ints. each list represents a turn.
    history = history[-(2 * config.max_history + 1):]
    print(f'Using this history: {history}')
    history_tokenized = [tokenizer.encode(utterance) for utterance in history]

    t0 = time.time()
    with torch.no_grad():
        print(f'Batch-sampling {config.num_samples} responses...')
        # out_ids is a list length num_samples of lists of ints
        finished_ids, unfinished_ids = sample_sequence(history_tokenized, tokenizer, model,
                                                       no_sample=config.no_sample,
                                                       max_length=config.max_length, min_length=config.min_length,
                                                       temperature=config.temperature,
                                                       top_k=config.top_k, top_p=config.top_p,
                                                       num_samples=config.num_samples, current_output=None)
        time_taken = time.time() - t0
        longest_sample = max([len(sample) for sample in finished_ids + unfinished_ids])
        print(f'Sampling {config.num_samples} samples (longest sample {longest_sample} tokens) took {time_taken} seconds')
    finished_responses = [tokenizer.decode(out, skip_special_tokens=True) for out in finished_ids]
    unfinished_responses = [tokenizer.decode(out, skip_special_tokens=True) for out in unfinished_ids]
    if unfinished_responses:
        print(f'Got some unfinished samples: {unfinished_responses}')
    print(f'Responses: {finished_responses}')

    return finished_responses


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        args.model_checkpoint = download_pretrained_model()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model, tokenizer = load_model(args.model, args.model_checkpoint, args.device)

    print('top_k: {}, top_p: {}, temperature: {}'.format(args.top_k, args.top_p, args.temperature))

    history = []
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids, _ = sample_sequence(history, tokenizer, model, args.min_length, args.max_length, args.device, args.temperature, args.top_p, args.top_k, args.no_sample)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)


if __name__ == "__main__":
    run()
