import torch
import torch.nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import numpy as np
import torch
from loguru import logger

import pandas as pd
import sys

def cal_log_prob(batch_caption):
    batch_tokens = [tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(x, add_prefix_space=True))
        for x in batch_caption]

    # add BOS and EOS
    tokenize_input = [[tokenizer.bos_token_id] + x + [tokenizer.eos_token_id]
            for x in batch_tokens]
    mask = []
    for sentence in tokenize_input:
        curr_mask = [1]*len(sentence) + [0] * (20-len(sentence))
        mask.append(curr_mask)
    mask = torch.FloatTensor(mask)
    for i in range(len(tokenize_input)):
        tokenize_input[i] = tokenize_input[i][: 20]
        tokenize_input[i].extend([tokenizer.eos_token_id] * (20 - len(tokenize_input[i]))) 
    tokenize_input = torch.LongTensor(tokenize_input)
    with torch.no_grad():
        outputs = model(tokenize_input, labels=tokenize_input, attention_mask = mask)
        loss, logits = outputs[:2]
        
    batch_lp = []
    for index in range(len(tokenize_input)):
        lp = 0.0
        for j in range(len(tokenize_input[index])):
            if mask[index][j]:
                masked_index = j
                predicted_score = logits[index][masked_index]
                predicted_prob = torch.nn.functional.softmax(predicted_score)
                lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_input[index][j]])[0]])
        batch_lp.append(lp)
    return batch_lp



tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')
captions = ['This is a sentence',
'A dog is barking at a man',
'Imminent change is required',
'A man with a red helmet is riding a bike']
cal_log_prob(captions)

logger.add(sys.stdout)