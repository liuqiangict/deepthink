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
        encodings = self.tokenizer.encode_plus(input_pairs, pad_to_max_length=True, max_length=self.max_seq_len)
        context_encodings = self.tokenizer.encode_plus(doc)

        start_idx, end_idx = self.get_correct_alignement(doc, answer)
        start_positions_context = context_encodings.char_to_token(start_idx)
        end_positions_context = context_encodings.char_to_token(end_idx-1)

        if start_positions_context is None or end_positions_context is None:
            print(encodings)
            start_positions, end_positions = 0, 0
        else:
            sep_idx = encodings['input_ids'].index(self.tokenizer.sep_token_id)
            start_positions = start_positions_context + sep_idx + 1
            end_positions = end_positions_context + sep_idx + 1
            if start_positions >= self.max_seq_len or end_positions >= self.max_seq_len:
                start_positions, end_positions = 0, 0

        #encodings.update({'start_positions': start_positions, 'end_positions': end_positions, 'attention_mask': encodings['attention_mask']})
   
        return encodings['input_ids'], encodings['attention_mask'], start_positions, end_positions

    def __getitem__(self, index):

        guid, query, doc, answer = self.data.all_pairs[index]
        input_ids, attention_mask, start_positions, end_positions = self.convert_to_features(query, doc, answer)

        #guid = map_to_torch([inputs['guid']])
        input_ids = map_to_torch(input_ids)
        attention_mask = map_to_torch(attention_mask)
        #token_type_ids = map_to_torch(inputs['token_type_ids'])
        start_positions = map_to_torch([start_positions])[0]
        end_positions = map_to_torch([end_positions])[0]
        #return {'id': guid, 'query': query, 'doc': doc, 'answers': {'text': [answer]}, 'input_ids': input_ids, 'attention_mask': attention_mask, 'start_positions': start_positions, 'end_positions': end_positions}
        return {'id': guid, 'query': query, 'doc': doc, 'answers': answer, 'input_ids': input_ids, 'attention_mask': attention_mask, 'start_positions': start_positions, 'end_positions': end_positions}

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
                doc = cols[2]
                answer = json.loads(cols[3])
                '''
                docs = [doc['Text'] for doc in json.loads(cols[2])]
                doc = ' '.join(docs)
                start_idx = int(cols[4])
                #end_idx = int(cols[5])
                answer_start = len(' '.join(docs[:start_idx]))
                if start_idx != 0:
                    answer_start += 1
                answer = {'text': [' '.join(docs[start_idx : end_idx + 1])], 'answer_start': [answer_start]}
                '''
                all_pairs.append([guid, query, doc, answer])
                
                if i > readin:
                    break
        
        #if mode == 'train':
        #    shuffle(all_pairs)
        self.all_pairs = all_pairs
        self.len = len(self.all_pairs)

    def __len__(self):
        return self.len


