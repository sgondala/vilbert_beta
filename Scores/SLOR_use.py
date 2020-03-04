import torch
import torch.nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import numpy as np
from loguru import logger
import sys
def cal_log_prob(caption):
    input_ids = torch.tensor(tokenizer.encode(caption, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    tokenize_input = tokenizer.tokenize(caption)
        #50256 is the token_id for <|endoftext|>
    tensor_input = torch.tensor([ [50256]  +  tokenizer.convert_tokens_to_ids(tokenize_input)])
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
log_prob_gt = []
with open('coco_description.json') as file:
    data = json.load(file)
    for i in range(len(data['images'])):
        gt = data['images'][i]['ground_truth']
        score = cal_log_prob(gt)
        log_prob_gt.append(score)

log_prob_predicted = []
with open('/srv/share3/rzhong34/updown-baseline/coco_train_prediction.json') as file:
    data = json.load(file)
    for i in range(len(data['annotations'])):
        cap = data['annotations'][i]['caption']
        score = cal_log_prob(cap)
        log_prob_gt.append(score)
import json
html_string = """
<font size = 40px face = 'Courier New'>"""    
from SLOR import score
from collections import Counter
#get the unigram log prob from the model
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


with open('coco_description.json') as caption_file:
    data = json.load(caption_file)
    temp = data['images']
    # data = captions_file['images']
    with open('/srv/share3/rzhong34/updown-baseline/coco_train_prediction.json') as file:
        other = json.load(file)
        table = "<table width='100%' border='0' cellpadding='10' cellspacing='0'>"
        tr = ""
        for i in range(0, 1000, 5):
            td = ""
            tr += "<tr>"
            for j in range(i, i + 5):
                td = ""
                image_url = temp[j]['url']
                gt_caption = temp[j]['caption']
                predicted = other[j]['caption']
                gt_score = score(unigram_prob, log_prob_gt[j], gt_caption)
                pred_score = score(unigram_prob, log_prob_predicted[j], predicted)
                td += "<td align='center' valign='center' >"
                td += "<img align='center' width='100%' height='auto' src={} />".format(image_url)
                td += "<br /><br />"
                td += "<b>{}</b>".format(gt_caption)
                td += "<br /><br />"
                td += "<b>GT: {}</b>".format(str(gt_score))
                td += "<br /><br />"
                td += "<b>{}</b>".format(predicted)
                td += "<br /><br />"
                td += "<b>Predicted: {}</b>".format(str(pred_score))
                td += "</td>"
                tr += td
            tr += "</tr>"
            if i % 100 == 0:
                print(i)
        table += tr
        table += "</table>"
        html_string += table
        


    
with open('coco_slor_GPT2.html', 'w') as f:
    f.write(html_string)

logger.add(sys.stdout)