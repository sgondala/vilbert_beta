import json
import numpy as np

html_high_actual = """
<font size = 40px face = 'Courier New'>"""
html_low_actual = """
<font size = 40px face = 'Courier New'>"""    

check = 'background '* 99 + 'background'

nocaps_metadata = json.loads(open('data/nocaps_val_image_info.json', 'r').readline())
nocaps_captions_metadata = json.load(open('data/nocaps_val_image_info_actual.json', 'r'))

open_image_id_to_captions = {}

for dict_entry in nocaps_captions_metadata:
    open_image_id = dict_entry['open_image_id']
    captions = dict_entry['captions']
    open_image_id_to_captions[open_image_id] = captions

image_id_to_open_image_id = {}
open_image_id_to_image_id = {}
for image_metadata in nocaps_metadata['images']:
    image_id = image_metadata['id']
    open_images_id = image_metadata['open_images_id']
    image_id_to_open_image_id[image_id] = open_images_id
    open_image_id_to_image_id[open_image_id] = image_id

def get_image_url_from_open_id(open_images_id):
    return 'https://s3.amazonaws.com/nocaps/val/' + open_images_id + '.jpg'

with open('results/nocaps_corr_both_clean_with_nocaps_df_ratio_5.json') as caption_file:
    data = json.load(caption_file)
    image_ids = data['image_ids']
    actual_cider_values = data['actual_values']
    predicted_cider_values = data['predicted_values']
    captions = data['captions']

    diff = np.array(actual_cider_values) - np.array(predicted_cider_values)
    low_actual_cider_indices = diff.argsort()[:100].tolist()
    high_actual_cider_indices = (-diff).argsort()[:100].tolist()
    all_indices = low_actual_cider_indices + high_actual_cider_indices

    table_high_actual = "<table width='100%' border='0' cellpadding='10' cellspacing='0'>"
    table_low_actual = "<table width='100%' border='0' cellpadding='10' cellspacing='0'>"
    tr_high_actual = ""
    tr_low_actual = ""

    # for i in range(0, 500, 2):
    for i in range(len(all_indices)):
        td = ""
        tr_high_actual += "<tr>"
        tr_low_actual += "<tr>"
        # for j in range(i, i + 2):
        for j in range(i, i+1):
            td = ""
            # print(i)
            index = all_indices[i]
            image_url = get_image_url_from_open_id(image_id_to_open_image_id[image_ids[index]])
            gt_caption = captions[index]
            predicted_cider = predicted_cider_values[index]
            actual_cider = actual_cider_values[index]
            actual_captions = open_image_id_to_captions[image_id_to_open_image_id[image_ids[index]]]
            td += "<td align='center' valign='center' >"
            td += "<img align='center' width='auto' height='150' src={} />".format(image_url)
            td += "<br /><br />"
            td += "<b>Generated caption: {}</b>".format(gt_caption)
            td += "<br /><br />"
            td += "<b>Predicted cider : {}</b>".format(predicted_cider)
            td += "<br /><br />"
            td += "<b>Actual cider : {}</b>".format(actual_cider)
            for k in range(len(actual_captions)):
                td += "<br /><br />"
                td += "<b>Actual caption : {}</b>".format(actual_captions[k])
            td += "<br /><br />"
            td += "</td>"
            if actual_cider - predicted_cider >= 0.5:
                tr_high_actual += td
            elif actual_cider - predicted_cider <= -0.5:
                tr_low_actual += td
            # tr += td
        tr_high_actual += "</tr>"
        tr_low_actual += "</tr>"
    table_high_actual += tr_high_actual
    table_low_actual += tr_low_actual
    table_high_actual += "</table>"
    table_low_actual += "</table>"
    
    html_high_actual += table_high_actual
    html_low_actual += table_low_actual
    
with open('nocaps_correlations_high_actual.html', 'w') as f:
    f.write(html_high_actual)

with open('nocaps_correlations_low_actual.html', 'w') as f:
    f.write(html_low_actual)