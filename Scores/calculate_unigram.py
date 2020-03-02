import json
from collections import Counter
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
