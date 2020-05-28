from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
      total += 1
      exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
      f1 += metric_max_over_ground_truths(
          f1_score, prediction, ground_truths)
    
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}



import torch
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering
from tqdm.auto import tqdm

tokenizer = LongformerTokenizerFast.from_pretrained('models')
model = LongformerForQuestionAnswering.from_pretrained('models')
model = model.cuda()
model.eval()

valid_dataset = torch.load('./data/valid_data.pt')
dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16)

answers = []
with torch.no_grad():
  for batch in tqdm(dataloader):
    start_scores, end_scores = model(input_ids=batch['input_ids'].cuda(),
                                  attention_mask=batch['attention_mask'].cuda())
    for i in range(start_scores.shape[0]):
      all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
      answer = ' '.join(all_tokens[torch.argmax(start_scores[i]) : torch.argmax(end_scores[i])+1])
      ans_ids = tokenizer.convert_tokens_to_ids(answer.split())
      answer = tokenizer.decode(ans_ids)
      answers.append(answer)

predictions = []
references = []
for ref, pred in zip(valid_dataset, answers):
  predictions.append(pred)
  references.append(ref['answers']['text'])

res = evaluate(references, predictions)
print(res)
