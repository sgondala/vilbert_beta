import torch
import torch.nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import numpy as np
import torch
from loguru import logger
from SLOR import score

import pandas as pd
import sys

import json
from collections import Counter
captions = []
with open('coco_annotations_2014/captions_train2014.json') as f:
    data = json.load(f)
    for ann in data['annotations']:
        captions.append(ann['caption'])
vocab = []
for cap in captions:
    if cap.endswith('.'):
        cap = cap[:-1]
    for i in cap.split():
        vocab.append(i.lower())
vocab_dict = Counter(vocab)
total = sum(vocab_dict.values())
unigram_prob = {k: v/total for k, v in vocab_dict.items()}


def cal_log_prob(caption):
    # batch_tokens = [tokenizer.convert_tokens_to_ids(
    #     tokenizer.tokenize(x, add_prefix_space=True))
    #     for x in batch_caption]

    # # add BOS and EOS
    # tokenize_input = [[tokenizer.bos_token_id] + x + [tokenizer.eos_token_id]
    #         for x in batch_tokens]
    # mask = []
    # for sentence in tokenize_input:
    #     curr_mask = [1]*len(sentence) + [0] * (20-len(sentence))
    #     mask.append(curr_mask)
    # mask = torch.FloatTensor(mask)
    # for i in range(len(tokenize_input)):
    #     tokenize_input[i] = tokenize_input[i][: 20]
    #     tokenize_input[i].extend([tokenizer.eos_token_id] * (20 - len(tokenize_input[i]))) 
    # tokenize_input = torch.LongTensor(tokenize_input)
    # with torch.no_grad():
    #     outputs = model(tokenize_input, labels=tokenize_input, attention_mask = mask)
    #     loss, logits = outputs[:2]
    #     print(logits.size())
    # batch_lp = []
    # for index in range(len(tokenize_input)):
    #     lp = 0.0
    #     for j in range(len(tokenize_input[index])):
    #         if mask[index][j]:
    #             masked_index = j
    #             predicted_score = logits[index][masked_index]
    #             predicted_prob = torch.nn.functional.softmax(predicted_score)
    #             lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_input[index][j]])[0]])
    #     batch_lp.append(float(lp))
    # return batch_lp

    tokenize_input = tokenizer.tokenize(caption)
        #50256 is the token_id for <|endoftext|>
    tensor_input = torch.tensor([ [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(tokenize_input) + [tokenizer.eos_token_id]])
    with torch.no_grad():
        outputs = model(tensor_input, labels=tensor_input)
        loss, logits = outputs[:2]
    lp = 0.0
    for i in range(len(tokenize_input)):
        masked_index = i
        predicted_score = logits[0, masked_index]
        predicted_prob = torch.nn.functional.softmax(predicted_score)
        lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_input[i]])[0]])
    return float(lp)




tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')
captions = [ "a white and white bird standing on a tree",
            "a bunch of fruits are sitting on a table",
            "a kitchen with a microwave in a kitchen",
            "a clock is sitting on top of a clock"]
scores = []
for i in captions:
    log_prob= (cal_log_prob(i))
    curr_score = score(unigram_prob, log_prob, i)
    scores.append(curr_score)
    

logger.add(sys.stdout)