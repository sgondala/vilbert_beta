import json
html_string = """
<font size = 40px face = 'Courier New'>"""    

check = 'background '* 99 + 'background'

with open('results/nocaps_dict_80_corr.json') as caption_file:
    data = json.load(caption_file)
    image_ids = data[]
    table = "<table width='100%' border='0' cellpadding='10' cellspacing='0'>"
    tr = ""
    for i in range(0, 1000, 2):
        td = ""
        tr += "<tr>"
        for j in range(i, i + 2):
            td = ""
            image_url = temp[j]['url']
            gt_caption = temp[j]['ground_truth']
            predicted = temp[j]['predicted']
            one_obj = temp[j]['one_obj_cat']
            objs = temp[j]['category']
            gt_score1 = objdesc(wvvecs1, vocablist1, objs, gt_caption)
            pred_score1 = objdesc(wvvecs1, vocablist1, objs, predicted)
            one_obj_score_gt1 = objdesc(wvvecs1, vocablist1, one_obj, gt_caption)
            one_obj_score_pred1 = objdesc(wvvecs1, vocablist1, one_obj, predicted)
            gt_score2 = objdesc(wvvecs2, vocablist2, objs, gt_caption)
            pred_score2 = objdesc(wvvecs2, vocablist2, objs, predicted)
            one_obj_score_gt2 = objdesc(wvvecs2, vocablist2, one_obj, gt_caption)
            one_obj_score_pred2 = objdesc(wvvecs2, vocablist2, one_obj, predicted)
            gt_score3 = objdesc(wvvecs3, vocablist3, objs, gt_caption)
            pred_score3 = objdesc(wvvecs3, vocablist3, objs, predicted)
            one_obj_score_gt3 = objdesc(wvvecs3, vocablist3, one_obj, gt_caption)
            one_obj_score_pred3 = objdesc(wvvecs3, vocablist3, one_obj, predicted)
            td += "<td align='center' valign='center' >"
            td += "<img align='center' width='auto' height='150' src={} />".format(image_url)
            td += "<br /><br />"
            td += '<b>{}</b>'.format(one_obj)
            td += "<br /><br />"
            td += "<b>Ground Truth: {}</b>".format(gt_caption)
            td += "<br /><br />"
            td += "<b>Predicted: {}</b>".format(predicted)
            td += "<table style='center'>"
            td += "<style>th, td {padding: 15px;}</style>"
            td += "<tr><th></th><th>glove</th><th>word2vec</th><th>fasttext</th></tr>"
            td += "<tr><th>GT repeated objs</th><th>{}</th><th>{}</th><th>{}</th></tr>".format(str(gt_score1), str(gt_score2), str(gt_score3))
            td += "<tr><th>Predicted repeated objs</th><th>{}</th><th>{}</th><th>{}</th></tr>".format(str(pred_score1), str(pred_score2), str(pred_score3))
            td += "<tr><th>GT no repeated objs</th><th>{}</th><th>{}</th><th>{}</th></tr>".format(str(one_obj_score_gt1), str(one_obj_score_gt2), str(one_obj_score_gt3))
            td += "<tr><th>Predicted no repeated objs</th><th>{}</th><th>{}</th><th>{}</th></tr>".format(str(one_obj_score_pred1), str(one_obj_score_pred2), str(one_obj_score_pred3))
            td += "</table>"
            td += "</td>"
            tr += td
        tr += "</tr>"
    table += tr
    table += "</table>"
    html_string += table
    
with open('nocaps_VIFIDEL.html', 'w') as f:
    f.write(html_string)