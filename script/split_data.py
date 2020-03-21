import json
import numpy as np
import random

coco_captions = json.load(open('../data/coco_generated_captions.json', 'r'))
coco_cider = json.load(open('../data/coco_cider_scores.json', 'r'))['CIDEr']

all_image_ids = [entry['image_id'] for entry in coco_captions]
all_image_ids = np.unique(all_image_ids)
random.shuffle(all_image_ids)

train_image_ids = all_image_ids[:19000]
test_image_ids = all_image_ids[19000:]

coco_train_captions = []
coco_test_captions = []
coco_train_cider = []
coco_test_cider = []

i = 0
for entry in coco_captions:
    if entry['image_id'] in train_image_ids:
        coco_train_captions.append(entry)
        coco_train_cider.append(coco_cider[i])
    else:
        coco_test_captions.append(entry)
        coco_test_cider.append(coco_cider[i])
    i += 1

assert len(coco_captions) == i

json.dump(coco_train_captions, open('../data/coco_train_captions_76000', 'w'))
json.dump(coco_test_captions, open('../data/coco_test_captions_4000', 'w'))

coco_train_cider_dict = {}
coco_train_cider_dict['CIDEr'] = coco_train_cider

coco_test_cider_dict = {}
coco_test_cider_dict['CIDEr'] = coco_test_cider

json.dump(coco_train_cider_dict, open('../data/coco_train_cider_76000', 'w'))
json.dump(coco_test_cider_dict, open('../data/coco_test_cider_4000', 'w'))