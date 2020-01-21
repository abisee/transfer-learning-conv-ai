"""
Loads utterances from csv file, generates responses under different settings, and writes to another csv file
"""
import logging
import random
import csv
import torch
import time
import json

from dataclasses import dataclass

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import add_special_tokens_
from interact import sample_sequence


INFILE = '/workspace/abi/transfer-learning-conv-ai/data/common_howwasyourday_responses.csv'
OUTFILE = '/workspace/abi/transfer-learning-conv-ai/data/common_howwasyourday_responses_answered.csv'

@dataclass
class DecodeConfig:
    # Default is natural sampling
    no_sample: bool = False
    min_length: int = 1
    max_length: int = 20
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0


def load_utterances(filepath):
    with open(filepath, 'r') as f:
        csvreader = csv.reader(f, delimiter=',')
        idx = 0
        rows = []
        for row in csvreader:
            if idx == 0:
                columns = row
            else:
                rows.append(row)
            idx += 1
    return rows, columns

def write_responses(filepath, columns, rows):
    print(f'writing to {filepath}')
    with open(filepath, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(columns)
        for row in rows:
            csvwriter.writerow(row)
    print('finished writing')


def get_config_string(config: dict):
    return ', '.join('{}={}'.format(key, val) for key, val in config.__dict__.items())


def main(model, model_checkpoint, configs, max_history=2, device="cuda" if torch.cuda.is_available() else "cpu", seed=42, num_samples=1):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info('Starting get_response')
    logger.info(f"device: {device}")

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

    rows, columns = load_utterances(INFILE)

    # Init outfile and write column names
    columns = ['freq', 'user_utterance']
    for config in configs:
        columns += ['RESPONSE ' + get_config_string(config), 'TIME_TAKEN ' + get_config_string(config)]
    with open(OUTFILE, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(columns)

    for row in rows:
        print()
        freq, utterance = row[0], row[1]
        history = ["how's your day going so far?", utterance]

        # Get history_tokenized, which should be a list of list of ints. each list represents a turn.
        history = history[-(2 * max_history + 1):]
        logger.info(f'Using this history: {history}')
        history_tokenized = [tokenizer.encode(utterance) for utterance in history]

        # Get responses for each config
        row = [freq, utterance]
        for config in configs:
            t0 = time.time()
            with torch.no_grad():
                logger.info(f'Batch-sampling {num_samples} responses...')
                # out_ids is a list length num_samples of lists of ints
                finished_ids, unfinished_ids = sample_sequence(history_tokenized, tokenizer, model, device=device, no_sample=config.no_sample,
                                                               max_length=config.max_length, min_length=config.min_length, temperature=config.temperature,
                                                               top_k=config.top_k, top_p=config.top_p, num_samples=num_samples, current_output=None)
                time_taken = time.time() - t0
                logger.info(f'Sampling {num_samples} samples (longest sample {max([len(sample) for sample in finished_ids+unfinished_ids])} tokens) took {time_taken} seconds')
            finished_responses = [tokenizer.decode(out, skip_special_tokens=True) for out in finished_ids]
            unfinished_responses = [tokenizer.decode(out, skip_special_tokens=True) for out in unfinished_ids]
            if unfinished_responses:
                logger.info(f'Got some unfinished samples: {unfinished_responses}')
            logger.info(f'Responses: {finished_responses}')
            row += [json.dumps(finished_responses), time_taken]

        # Append to outfile
        with open(OUTFILE, 'a') as f:
            csvwriter = csv.writer(f, delimiter=',')
            csvwriter.writerow(row)



if __name__ == "__main__":
    configs = [
        DecodeConfig(),  # natural sampling
        DecodeConfig(temperature=0.7, top_p=0.9),  # recommended settings
        DecodeConfig(temperature=0.7),  # should be less generic than recommended settings
        DecodeConfig(temperature=1.0, top_p=0.9),  # should be less generic than recommended settings
    ]
    main(model='gpt2-medium', model_checkpoint='runs/Jan04_22-40-10_ip-172-31-71-210_gpt2-medium', configs=configs,
         num_samples=10)