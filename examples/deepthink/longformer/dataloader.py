import argparse
import csv
import logging
import os
import random
import sys
import json
import datetime
from itertools import product

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.distributed as dist
import torch.nn.functional as F
from random import shuffle
from typing import Tuple

def map_to_torch(encoding):
    encoding = torch.LongTensor(encoding)
    encoding.requires_grad_(False)
    return encoding

def map_to_torch_float(encoding):
    encoding = torch.FloatTensor(encoding)
    encoding.requires_grad_(False)
    return encoding


class DTDataset(Dataset):
    def __init__(self, tokenizer, data, max_seq_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = max_seq_len
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def get_correct_alignement(self, context, answer):
        """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
        #gold_text = answer
        #start_idx = answer_start
        gold_text = answer['text'][0]
        start_idx = answer['answer_start'][0]
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx       # When the gold label position is good
        elif context[start_idx - 1 : end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1   # When the gold label is off by one character
        elif context[start_idx - 2 : end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2   # When the gold label is off by two character
        else:
            raise ValueError(f'Incorrect answer as {gold_text} to span from text as \'{context[start_idx:end_idx]}\'')

    def convert_to_features(self, query, doc, answer):
        input_pairs = [query, doc]
        encodings = self.tokenizer.encode_plus(input_pairs, pad_to_max_length=True, max_length=self.max_seq_len, return_token_type_ids=True, return_attention_mask=True)

        return encodings['input_ids'], encodings['global_attention_mask'], encodings['attention_mask'], encodings['token_type_ids'], encodings['valid_mask_ids'],  encodings['label']


    def __getitem__(self, index):
        guid, query, docs, label = self.data.all_pairs[index]
        encodings = self.tokenizer.encode_plus(query, docs, label, pad_to_max_length=True, max_length=self.max_seq_len, return_token_type_ids=True, return_attention_mask=True)

        input_ids = map_to_torch(encodings['input_ids'])
        global_attention_mask = map_to_torch(encodings['global_attention_mask'])
        attention_mask = map_to_torch(encodings['attention_mask'])
        token_type_ids = map_to_torch(encodings['token_type_ids'])
        valid_mask_ids = map_to_torch(encodings['valid_mask_ids'])
        positions = map_to_torch([encodings['label']])[0]
        return {'id': guid, 'query': query, 'docs': docs, 'label': label, 'input_ids': input_ids, 'global_attention_mask': global_attention_mask, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids, 'valid_mask_ids': valid_mask_ids, 'start_positions': positions}

class DeepThinkDataset:
    def __init__(self, path, readin=200000000, mode='train'):
        all_pairs = []
        with open(path, encoding="utf-8") as fd:
            for i, line in enumerate(tqdm(fd)):
                cols = line.strip().split('\t')
                if len(cols) != 6:
                    continue
                guid = int(cols[0])
                query = cols[1]
                if len(query) > 512:
                    continue
                docs = [doc['Text'] for doc in json.loads(cols[2])]
                label = int(cols[4])

                all_pairs.append([guid, query, docs, label])
                
                if i > readin:
                    break
        
        #if mode == 'train':
        #    shuffle(all_pairs)
        self.all_pairs = all_pairs
        self.len = len(self.all_pairs)

    def __len__(self):
        return self.len


