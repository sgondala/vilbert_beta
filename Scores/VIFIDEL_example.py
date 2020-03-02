from scorer import objdesc
from scorer import preprocess
from collections import Counter

wvvecs1, vocablist1 = preprocess('glove.840B.300d.word2vec.txt', False, 'glove')

with open('nocaps_description.json') as caption_file:
    data = json.load(caption_file)
    temp = data['images']
    for i in range(0, len(temp), 2):
        td = ""
        tr += "<tr>"
        for j in range(i, i + 2):
            image_url = temp[j]['url']
            gt_caption = temp[j]['ground_truth']
            good_predicted = temp[j]['good_predicted']
            bad_predicted = temp[j]['bad_predicted']
            objs = temp[j]['category'] # ground truth annotations
            gt_score1 = objdesc(wvvecs1, vocablist1, objs, gt_caption)
            good_pred_score1 = objdesc(wvvecs1, vocablist1, objs, good_predicted)
            bad_pred_score1 = objdesc(wvvecs1, vocablist1, objs, bad_predicted)
            gt_glove.append(gt_score1)
            good_glove.append(good_pred_score1)
            bad_glove.append(bad_pred_score1)

        
