import collections
from collections import defaultdict
import textblob
import csv
from textblob import TextBlob
import cPickle as pickle 

# def default_():
#  return 0

# count_dict = defaultdict(default_)
count_dict = {}

i = 0
with open('../data/Train_GCC-training.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        if i % 1000 == 0:
            print(i)
        caption = row[0].lower().decode('ascii', errors="ignore")
        try:
            unigrams = TextBlob(caption).ngrams(1)
        except:
            print(caption)
            assert False
        bigrams = TextBlob(caption).ngrams(2)
        trigrams = TextBlob(caption).ngrams(3)
        fourgrams = TextBlob(caption).ngrams(4)
        allgrams = []
        allgrams += unigrams + bigrams + trigrams + fourgrams
        for value in allgrams:
            value = tuple(value)
            if value in count_dict: 
                count_dict[value] += 1
            else:
                count_dict[value] = 1
        i += 1

pickle.dump(count_dict, open('conceptual-caps-val-df.p', 'w'))
