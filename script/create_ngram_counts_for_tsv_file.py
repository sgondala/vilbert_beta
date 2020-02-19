import collections
from collections import defaultdict
import textblob
import csv
from textblob import TextBlob

count_dict = defaultdict(lambda: 0)

i = 0
with open('myfile.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        if i > 100:
            break
        print(i)
        caption = row[0].lower()
        unigrams = TextBlob(caption).ngrams(1)
        bigrams = TextBlob(caption).ngrams(2)
        trigrams = TextBlob(caption).ngrams(3)
        fourgrams = TextBlob(caption).ngrams(4)
        allgrams = []
        allgrams += unigrams + bigrams + trigrams + fourgrams
        for value in allgrams:
            count_dict[tuple(value)] += 1
        i += 1

print(count_dict)

