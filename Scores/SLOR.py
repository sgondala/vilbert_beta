

import numpy as np

from nltk.tokenize import word_tokenize
from typing import Any, Dict, List, Tuple, Union
PUNCTUATIONS: List[str] = [
    "''", "'", "``", "`", "(", ")", "{", "}",
    ".", "?", "!", ",", ":", "-", "--", "...", ";", '$']
# def score(unigram_prob_dict, log_prob, captions):
#     i = 0
#     batch_score = []
#     for caption in captions:
#         caption_tokens: List[str] = word_tokenize(caption)
#         caption_tokens = [ct.lower() for ct in caption_tokens if ct not in PUNCTUATIONS]
#         print(caption_tokens)

#         curr_uni = 0.0
#         for token in caption_tokens:
#             if token not in unigram_prob_dict.keys():
#                 return -1 ## change this handling of oov?
#             curr_uni += np.log(unigram_prob_dict[token])
#         print(len(caption_tokens))
#         score = (log_prob[i] - curr_uni) / len(caption_tokens)
        
#         batch_score.append(score)
#         i += 1

#     return batch_score
def score(unigram_prob_dict, logprob, caption):
    caption_tokens: List[str] = word_tokenize(caption)
    caption_tokens = [ct.lower() for ct in caption_tokens if ct not in PUNCTUATIONS]

    curr_uni = 0.0
    for token in caption_tokens:
        if token not in unigram_prob_dict.keys():
            return -1 ## change this handling of oov?
        curr_uni += np.log(unigram_prob_dict[token])
    n = len(caption.split())

    return ((logprob - curr_uni) / n)