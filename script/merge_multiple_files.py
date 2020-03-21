import json
import numpy as np
import random

coco_train_captions = json.load(open('../data/coco_train_captions_76000.json', 'r'))
nocaps_train_captions = json.load(open('../data/nocaps_train_data_cleaned_2400.json', 'r'))

coco_cider = json.load(open('../data/coco_train_cider_76000.json', 'r'))['CIDEr']
nocaps_cider = json.load(open('../data/nocaps_cider_train_data_cleaned_2400.json', 'r'))['CIDEr']

combined_captions = coco_train_captions + nocaps_train_captions
combined_cider = coco_cider + nocaps_cider

combined_cider_dict = {}
combined_cider_dict['CIDEr'] = combined_cider

json.dump(combined_captions, open('../data/combined_coco_and_nocaps_all_clean_train.json', 'w'))
json.dump(combined_cider, open('../data/combined_coco_and_nocaps_all_clean_cider_train.json', 'w'))