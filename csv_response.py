"""
Loads utterances from csv file, generates responses under different settings, and writes to another csv file
"""
import logging
import random
import csv
import torch
import json

from interact import sample_sequence, load_model, batch_decode, DecodeConfig


INFILE = '/workspace/abi/transfer-learning-conv-ai/data/common_howwasyourday_responses.csv'
OUTFILE = '/workspace/abi/transfer-learning-conv-ai/data/common_howwasyourday_responses_answered.csv'


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


def main(model, model_checkpoint, configs, device="cuda" if torch.cuda.is_available() else "cpu", seed=42):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info('Starting get_response')
    logger.info(f"device: {device}")

    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model, tokenizer = load_model(model, model_checkpoint, device)

    rows, columns = load_utterances(INFILE)

    # Init outfile and write column names
    columns = ['freq', 'user_utterance']
    for config in configs:
        columns += ['RESPONSE ' + config.__str__]
    with open(OUTFILE, 'w') as f:
        csvwriter = csv.writer(f, delimiter=',')
        csvwriter.writerow(columns)

    for row in rows:
        print()
        freq, utterance = row[0], row[1]
        history = ['how\'s your day going so far?', utterance]

        # Init row to write to file
        row_to_write = [freq, utterance]

        # Get responses for each config
        for config in configs:
            finished_responses = batch_decode(model, tokenizer, history, config)
            row_to_write += [json.dumps(finished_responses)]

        # Append to outfile
        with open(OUTFILE, 'a') as f:
            csvwriter = csv.writer(f, delimiter=',')
            csvwriter.writerow(row_to_write)



if __name__ == "__main__":
    configs = [
        DecodeConfig(),  # natural sampling
        DecodeConfig(temperature=0.7, top_p=0.9),  # recommended settings
        DecodeConfig(temperature=0.7),  # should be less generic than recommended settings
        DecodeConfig(temperature=1.0, top_p=0.9),  # should be less generic than recommended settings
    ]
    main(model='gpt2-medium', model_checkpoint='runs/Jan04_22-40-10_ip-172-31-71-210_gpt2-medium', configs=configs)