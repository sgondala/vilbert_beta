import json
html_string = """
<font size = 40px face = 'Courier New'>"""    

check = 'background '* 99 + 'background'

nocaps_metadata = json.loads(open('data/nocaps_val_image_info.json', 'r').readline())
image_id_to_open_image_id = {}
for image_metadata in nocaps_metadata['images']:
    image_id = image_metadata['id']
    open_images_id = image_metadata['open_images_id']
    image_id_to_open_image_id[image_id] = open_images_id

def get_image_url_from_open_id(open_images_id):
    return 'https://s3.amazonaws.com/nocaps/val/' + open_images_id + '.jpg'

with open('results/nocaps_dict_80_corr.json') as caption_file:
    data = json.load(caption_file)
    image_ids = data['image_ids']
    actual_cider_values = data['actual_values']
    predicted_cider_values = data['predicted_values']
    captions = data['captions']

    table = "<table width='100%' border='0' cellpadding='10' cellspacing='0'>"
    tr = ""
    for i in range(0, 500, 2):
        td = ""
        tr += "<tr>"
        for j in range(i, i + 2):
            td = ""
            image_url = get_image_url_from_open_id(image_id_to_open_image_id[image_ids[i]])
            gt_caption = captions[i]
            predicted_cider = predicted_cider_values[i]
            actual_cider = actual_cider_values[i]
            td += "<td align='center' valign='center' >"
            td += "<img align='center' width='auto' height='150' src={} />".format(image_url)
            # td += "<br /><br />"
            # td += '<b>{}</b>'.format(one_obj)
            td += "<br /><br />"
            td += "<b>Ground Truth: {}</b>".format(gt_caption)
            td += "<br /><br />"
            td += "<b>Predicted cider : {}</b>".format(predicted_cider)
            td += "<br /><br />"
            td += "<b>Actual cider : {}</b>".format(actual_cider)
            # td += "<table style='center'>"
            # td += "<style>th, td {padding: 15px;}</style>"
            # td += "<tr><th></th><th>glove</th><th>word2vec</th><th>fasttext</th></tr>"
            # td += "<tr><th>GT repeated objs</th><th>{}</th><th>{}</th><th>{}</th></tr>".format(str(gt_score1), str(gt_score2), str(gt_score3))
            # td += "<tr><th>Predicted repeated objs</th><th>{}</th><th>{}</th><th>{}</th></tr>".format(str(pred_score1), str(pred_score2), str(pred_score3))
            # td += "<tr><th>GT no repeated objs</th><th>{}</th><th>{}</th><th>{}</th></tr>".format(str(one_obj_score_gt1), str(one_obj_score_gt2), str(one_obj_score_gt3))
            # td += "<tr><th>Predicted no repeated objs</th><th>{}</th><th>{}</th><th>{}</th></tr>".format(str(one_obj_score_pred1), str(one_obj_score_pred2), str(one_obj_score_pred3))
            # td += "</table>"
            td += "</td>"
            tr += td
        tr += "</tr>"
    table += tr
    table += "</table>"
    html_string += table
    
with open('nocaps_correlations.html', 'w') as f:
    f.write(html_string)