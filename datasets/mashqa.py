import os
import json
import pandas as pd

processed_files = {'train': "/Users/namrata/Downloads/Fall 2022/Healthcare/project/dataset/mashqa_data/processed/train.json",
'val': "/Users/namrata/Downloads/Fall 2022/Healthcare/project/dataset/mashqa_data/processed/val.json",
'test': "/Users/namrata/Downloads/Fall 2022/Healthcare/project/dataset/mashqa_data/processed/test.json"}

data_files = {'train': "/Users/namrata/Downloads/Fall 2022/Healthcare/project/dataset/mashqa_data/train_webmd_squad_v2_full.json",
'val': "/Users/namrata/Downloads/Fall 2022/Healthcare/project/dataset/mashqa_data/processed/val_webmd_squad_v2_full.json",
'test': "/Users/namrata/Downloads/Fall 2022/Healthcare/project/dataset/mashqa_data/processed/test_webmd_squad_v2_full.json"}

def process(r, file_name):
    input_data = r['data']
    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            sent_starts = paragraph['sent_starts']
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                orig_answer_text = None
                is_impossible = False
                start_poses=[]
    #                 if is_training:
                is_impossible = qa["is_impossible"]
                if (len(qa["answers"]) != 1) and (not is_impossible):
                    raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                if not is_impossible:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    start_position = answer["answer_start"]
                    start_poses = answer["answer_starts"]
                else:
                    start_position = -1
                    orig_answer_text = ""
                example = {'qas_id': qas_id, 'question': question_text, 'context': paragraph_text, 'answer': orig_answer_text, 'is_impossible': is_impossible}
                examples.append(example)

    print(examples[:5])
    json.dump(examples, open(file_name, 'w'))

for i, data_file in enumerate(data_files):
    r = json.load(open(data_file, 'r'))
    process(r, processed_files[i])

