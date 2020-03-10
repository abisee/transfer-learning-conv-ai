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

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset_personalities, download_pretrained_model


# Default is the recommended settings in transfer-learning-conv-ai
# max_history_tokens=800 is a rough max based on gpt2's max size of 1024 (plus leaving space for speaker tokens and generated response)
# See here: https://github.com/huggingface/transformers/issues/1749
DEFAULT_DECODE_CONFIG = {
    'no_sample': False,
    'min_length': 1,
    'max_length': 20,
    'temperature': 0.7,
    'top_k': 0,
    'top_p': 0.9,
    'max_history_tokens': 800,
    'num_samples': 10,
    'response_prefix': '',
}


def complete_config(config):
    """
    Fill in any missing keys in config with the value in DEFAULT_DECODE_CONFIG. If we find any keys in config that
    aren't in DEFAULT_DECODE_CONFIG, raise an error.
    """
    # for key, val in config.items():
    #     if key not in DEFAULT_DECODE_CONFIG:
    #         raise ValueError(f'Unrecognized key "{key}" in config: {config}. Valid keys are: {list(DEFAULT_DECODE_CONFIG.keys())}')
    for key, val in DEFAULT_DECODE_CONFIG.items():
        if key not in config:
            config[key] = val
    return config


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
                    num_samples=1, current_output=[]):
    """
    Inputs:
        history: The conversation history. List (odd length) of utterances, where the first utterance is the user's.
            Each utterance is a list of ints (the tokenized utterance).
        tokenizer: the gpt2 tokenizer
        model: the gpt2 model
        device: str. 'cuda' or 'cpu'
        no_sample: bool. if False, use greedy decoding
        max_length: int. maximum length (in gpt2 tokens) of the generated output
        min_length: int. minimum length (in gpt2 tokens) of the generated output
        temperature: float. softmax temperature
        top_k: int. k for top k decoding
        top_p: float. p for top p decoding
        num_samples: int. The number of samples we want to generate for the response
        current_output: list of ints. The prefix you want the response to begin with (tokenized).

    Returns:
        finished_outputs: list (length <= num_samples) of lists (min_length <= length <= max_length) of ints.
            These are utterances where the model predicted the end token.
        unfinished_outputs: list (length <= num_samples) of lists (length max_length) of ints.
            These are utterances where we reached max_length before the model predicted the end token (hence unfinished)
    """

    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    # Initialize current_outputs, which will accumulate the generated output for each sample
    current_outputs = [[i for i in current_output] for _ in range(num_samples)]

    # Record whether each sample is finished or not
    finished = [False for _ in range(num_samples)]

    past = None

    for i in range(max_length):
        # print(f'step {i} of max {max_length}')

        # Get input_ids and token_type_ids, both tensors of shape (num_samples, seqlen).
        if i == 0:
            # On the first turn seqlen is the history length (plus current_output length).
            instances = [build_input_from_segments(history, current_output, tokenizer, with_eos=False) for current_output in current_outputs]
            input_ids = torch.tensor([instance["input_ids"] for instance in instances], device=device)
            token_type_ids = torch.tensor([instance["token_type_ids"] for instance in instances], device=device)
        else:
            # On subsequent turns seqlen=1 because we are using past.
            input_ids = torch.tensor([[current_output[-1]] for current_output in current_outputs], device=device)
            speaker2 = tokenizer.convert_tokens_to_ids('<speaker2>')
            token_type_ids = torch.tensor([[speaker2] for _ in current_outputs], device=device)

        # Get predictions from the model
        logits, past = model(input_ids, token_type_ids=token_type_ids, past=past)
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
    print("Getting pretrained model and tokenizer (device={})...".format(device))
    tokenizer_class = GPT2Tokenizer if "gpt2" in model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" in model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(model_checkpoint)
    model.to(device)
    add_special_tokens_(model, tokenizer)
    return model, tokenizer


def tokenize_truncate_history(history, tokenizer, config):
    """Tokenize history, then truncate it as necessary to fit under config['max_history_tokens']"""
    assert len(history) % 2 == 1, f'ERROR: history should have odd number of utterances (with user going first), but this history has ({len(history)}): {history}'

    history_tokenized = [tokenizer.encode(utterance) for utterance in history]  # list of list of ints
    history_lens = [len(utt) for utt in history_tokenized]

    # Get the longest odd-length history (i.e. starts with user) that fits under max_history_tokens
    for num_utts in range(len(history), 0, -2):
        trunc_history_lens = history_lens[-num_utts:]

        # See if the truncated version is short enough
        if sum(trunc_history_lens) <= config['max_history_tokens']:

            # If we're truncating, log
            if num_utts != len(history):
                print('Full history ({} utterances / {} tokens) is longer than max_history_tokens={}. Truncating to last {} utterances ({} tokens)'.format(
                    len(history), sum(history_lens), config['max_history_tokens'], num_utts, sum(trunc_history_lens)))

            history_tokenized = history_tokenized[-num_utts:]
            history = history[-num_utts:]
            break

        # If just the last single utterance is too long, raise error
        elif num_utts == 1:
            raise ValueError(f'The last utterance {history[-1]} is too long ({history_lens[-1]} tokens) to fit under the required max_history_tokens={config["max_history_tokens"]}')

    assert len(history) == len(history_tokenized)
    return history, history_tokenized


def batch_decode(model, tokenizer, history, config):
    """
    Generate num_samples responses to history.

    @param model:
    @param tokenizer:
    @param history: list of strings (each one an utterance)
    @param config: dict
    @return: finished_responses: list (length <=config['num_samples']) of strings, each a response to history.
    """

    # Tokenize and truncate history
    history_used, history_tokenized = tokenize_truncate_history(history, tokenizer, config)

    # Get current_output (i.e. the start of the response) if applicable
    if config['response_prefix']:
        current_output = tokenizer.encode(config['response_prefix'])  # list of ints
    else:
        current_output = []

    # Log
    print('Generating {} responses to this history ({} utterances / {} tokens): {} with this config: {}'.format(
        config["num_samples"], len(history_used), sum([len(utt) for utt in history_tokenized]), history_used, config))

    t0 = time.time()
    with torch.no_grad():
        # out_ids is a list length num_samples of lists of ints
        finished_ids, unfinished_ids = sample_sequence(history_tokenized, tokenizer, model,
                                                       no_sample=config['no_sample'],
                                                       max_length=config['max_length'], min_length=config['min_length'],
                                                       temperature=config['temperature'],
                                                       top_k=config['top_k'], top_p=config['top_p'],
                                                       num_samples=config['num_samples'], current_output=current_output)
        time_taken = time.time() - t0
        longest_sample = max([len(sample) for sample in finished_ids + unfinished_ids])
        print('Finished generating. Getting {} samples (longest sample {} tokens) took {} seconds'.format(config["num_samples"], longest_sample, time_taken))
    finished_responses = [tokenizer.decode(out, skip_special_tokens=True) for out in finished_ids]
    unfinished_responses = [tokenizer.decode(out, skip_special_tokens=True) for out in unfinished_ids]
    if unfinished_responses:
        print('Got some unfinished samples: {}'.format(unfinished_responses))
    print('Responses: {}'.format(finished_responses))

    return finished_responses, unfinished_responses, history_used


# def run():
#     parser = ArgumentParser()
#     parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
#     parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
#     parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
#     parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
#     parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
#
#     parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
#     parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
#     parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
#     parser.add_argument("--seed", type=int, default=42, help="Seed")
#     parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
#     parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
#     parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
#     args = parser.parse_args()
#
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__file__)
#     logger.info(pformat(args))
#
#     if args.model_checkpoint == "":
#         args.model_checkpoint = download_pretrained_model()
#
#     random.seed(args.seed)
#     torch.random.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)
#
#     model, tokenizer = load_model(args.model, args.model_checkpoint, args.device)
#
#     print('top_k: {}, top_p: {}, temperature: {}'.format(args.top_k, args.top_p, args.temperature))
#
#     history = []
#     while True:
#         raw_text = input(">>> ")
#         while not raw_text:
#             print('Prompt should not be empty!')
#             raw_text = input(">>> ")
#         history.append(tokenizer.encode(raw_text))
#         with torch.no_grad():
#             out_ids, _ = sample_sequence(history, tokenizer, model, args.min_length, args.max_length, args.device, args.temperature, args.top_p, args.top_k, args.no_sample)
#         history.append(out_ids)
#         history = history[-(2*args.max_history+1):]
#         out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
#         print(out_text)



def run_remote_module():
    """for interactive testing on the cluster"""

    seed = 1
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load model
    MODEL = 'gpt2-medium'
    MODEL_CHECKPOINT = 'runs/Jan04_22-40-10_ip-172-31-71-210_gpt2-medium'
    model, tokenizer = load_model(MODEL, MODEL_CHECKPOINT)

    msg = {
        'history': ['i am having such a bad day today!', 'oh no! why is that?', 'i fell down a well'],
        'config': {}
    }

    history = msg['history']
    config = msg['config'] if 'config' in msg else {}  # dict
    config = complete_config(config)
    responses, unfinished_responses, history_used = batch_decode(model, tokenizer, history, config)


if __name__ == "__main__":
    # run()
    run_remote_module()
