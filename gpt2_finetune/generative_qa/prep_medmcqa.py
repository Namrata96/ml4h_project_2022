import pandas as pd
import json
import numpy as np
import os

root_dir = "/scratch/as14770/ML4H/ml4h_project_2022"
train_json = os.path.join(root_dir, "Data/MedMCQA/medmcqa_train.json")
val_json = os.path.join(root_dir, "Data/MedMCQA/dev.json")
test_json = os.path.join(root_dir, "Data/MedMCQA/test.json")

train_json_data = pd.read_json(train_json, lines = True)

def format_for_abstractive_qa(train_json_data):
    formatted_train_df = []
    for idx, entry in train_json_data.iterrows():
        try:
            c_idx = int(entry['cop'])-1
            if c_idx > 3 or c_idx < 0:
                return np.nan
            answer = [entry['opa'],entry['opb'],entry['opc'],entry['opd']][c_idx]
            formatted_entry = {'exp': entry['exp'], 'question': entry['question'], 'answer': answer, 'input_text': 'Context: ' + entry['exp'] + '\nQuestion: ' + entry['question']}
        except Exception:
            formatted_entry = {'exp': None, 'question': None, 'answer': None, 'input_text': None}
        formatted_train_df.append(formatted_entry)
    return formatted_train_df

formatted_train_df = pd.DataFrame(format_for_abstractive_qa(train_json_data))

formatted_train_df.to_csv(os.path.join(root_dir, 'Data/MedMCQA/train_for_abstractive_qa.csv'))
