import pandas as pd
import json
import numpy as np
import os

root_dir = "/scratch/as14770/ML4H/ml4h_project_2022"
train_json = os.path.join(root_dir, "Data/MedMCQA/medmcqa_train.json")
val_json = os.path.join(root_dir, "Data/MedMCQA/dev.json")
test_json = os.path.join(root_dir, "Data/MedMCQA/test.json")

train_json_data = pd.read_json(train_json, lines = True)

def format_for_extractive_qa(train_json_data):
    formatted_train_df = []
    for idx, entry in train_json_data.iterrows():
        try:
            c_idx = int(entry['cop'])-1
            if c_idx > 3 or c_idx < 0:
                return np.nan
            answer = [entry['opa'],entry['opb'],entry['opc'],entry['opd']][c_idx]
            answer_start_idx = entry['exp'].find(answer)
            formatted_text = {'id': idx, 
                              'title': entry['question'],
                              'context': entry['exp'],
                              'question': entry['question'],
                              'answers': {'text': [answer], 'answer_start': [answer_start_idx]}}
        except Exception:
            formatted_text = {'id': None, 
                              'title': None,
                              'context': None,
                              'question': None,
                              'answers': {'text': [None], 'answer_start': [None]}}
        formatted_train_df.append(formatted_text)
    return formatted_train_df

formatted_train_df = pd.DataFrame(format_for_extractive_qa(train_json_data))

formatted_train_df.dropna(inplace = True)

train_df = formatted_train_df.sample(frac = 0.9, random_state = 100)
val_df = formatted_train_df.drop(train_df.index)

train_df.to_csv(os.path.join(root_dir, 'Data/MedMCQA/train_for_extractive_qa.csv'))
val_df.to_csv(os.path.join(root_dir, 'Data/MedMCQA/val_for_extractive_qa.csv'))