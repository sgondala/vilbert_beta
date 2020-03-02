
import numpy as np

##
## Unigram_prob: a dictionary that contains all the vocabulary; use coco corpus
## Log probability of the sentence evaluated using the language model
## Sentence: predicted caption
## returns -1 if there are unknown words
def score(unigram_prob, logprob, sentence):
    uni = 0.0
    if sentence.endswith('.'):
        sentence = sentence[:-1]
    for w in sentence.split():
        if not w.lower()[-1].isalpha():
            w = w.lower()[:-1]
        if w.lower() not in unigram_prob.keys():
            return -1
        uni += np.log(unigram_prob[w.lower()])
    n = len(sentence.split())
    return ((logprob - uni) / n)

