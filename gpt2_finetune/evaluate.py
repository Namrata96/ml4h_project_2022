import os
import json
import sys
import pandas as pd
import numpy as np
import torch
device = torch.device("cuda")
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
# from inference_dataset import GPT2QAInferenceDataset
sys.path.append("/scratch/as14770/ML4H/")
# from ml4h_project_2022.metrics import rouge, bleu, mcc

root_dir = "/scratch/as14770/ML4H/ml4h_project_2022"
path_to_tuned_weights = "gpt2_finetune/mcq_w_exp_qa/mcq_qa_letter_opt_gpt2_finetuned_2.pth"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token = '<|pad|>')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load(os.path.join(root_dir, path_to_tuned_weights)))
model.to(device)
model.eval()

dataset_path = "validation_set_tested_gpt3.json"
with open(dataset_path) as data_file:
    json_data = json.load(data_file)

data_df = pd.DataFrame(json_data)

results_dict = {"id": [], 
                "prediction_text": [], 
                "prediction": [], 
                "correct_answer": [], 
                "label": [], 
                "binary_correct": [], 
                "subject_name": []}

for idx, entry in data_df.iterrows():
    results_dict["id"].append(entry['id'])
    if entry['exp'] is None:
        entry['exp'] = ""
    else:
        exp = entry['exp']
        entry['exp'] = exp.replace([entry['opa'], entry['opb'], entry['opc'], entry['opd']][entry['cop']], "")
    prompt = 'Question: ' + entry['question'] + '\nOptions: ' + "\nA)" + entry['opa']  + "\nB)" + entry['opb'] + "\nC)" + entry['opc'] + "\nD)" + entry['opd'] + '\nAnswer: Let\'s derive the differential diagnosis step by step'
#     if entry['exp'] is not None:
#         prompt = 'Context: ' + entry['exp'] + '\nQuestion: ' + entry['question'] + '\nOptions: ' + "\n(A)" + entry['opa']  + "\n(B)" + entry['opb'] + "\n(C)" + entry['opc'] + "\n(D)" + entry['opd'] + '\nAnswer: '
    prompt_tokenized = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    if prompt_tokenized.shape[-1] >= 1024:
        prompt = 'Question: ' + entry['question'] + '\nOptions: ' + "\nA)" + entry['opa']  + "\nB)" + entry['opb'] + "\nC)" + entry['opc'] + "\nD)" + entry['opd'] + '\nAnswer: Let\'s derive the differential diagnosis step by step'
    print("Prompt: \n", prompt)
    prompt_tokenized = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    prompt_tokenized = prompt_tokenized.to(device)
    sample_output = model.generate(prompt_tokenized,                                
                                   max_new_tokens = 25)
    predicted_answer = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    predicted_answer = predicted_answer.replace(prompt, "")
    print("\nPredicted answer: \n", predicted_answer)
    if len(predicted_answer.split("\n")) >= 2:
        predicted_answer = predicted_answer.split("\n")[-2]
    else:
        predicted_answer = predicted_answer
    predicted_answer = predicted_answer.replace("Answer: ", "")
    pred_option_letter = None
    for opt_idx, opt in enumerate([entry['opa'], entry['opb'], entry['opc'], entry['opd']]):
        if opt.lower() == predicted_answer.lower():
            pred_option_letter = ['A', 'B', 'C', 'D'][opt_idx]
            break
    results_dict["prediction"].append(pred_option_letter)
    results_dict["prediction_text"].append(predicted_answer)
    corr_idx = entry['cop']
    corr_answer = [entry['opa'], entry['opb'], entry['opc'], entry['opd']][corr_idx]
    corr_option_letter = ['A', 'B', 'C', 'D'][corr_idx]
    results_dict["correct_answer"].append(corr_answer)
    results_dict["label"].append(corr_option_letter)
    results_dict["subject_name"].append(entry["subject_name"])
    if pred_option_letter == corr_option_letter:
        binary_correct = 1
    else:
        binary_correct = 0
    results_dict["binary_correct"].append(binary_correct)
    print("\nCorrect Answer: (" + corr_option_letter + ')' + corr_answer)
    print("\nNEXT\n")
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("gpt2_ft_eval_exp_letter_1_processed_modified_letter_opt_trained.csv")



    
