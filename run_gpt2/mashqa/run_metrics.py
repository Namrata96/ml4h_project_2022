
import json
import random
from operator import itemgetter 

import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification

from transformers import pipeline
from evaluate import load
import torch
from metrics import *

def sampled_evaluation(prediction_file:str, sample_ratio:int):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Running sampled evaluation")
    random.seed(42)
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    ids = set()
    for prompt in predictions:
        for prediction in predictions[prompt]:
            ids.insert(prediction['qas_id'])
    
    num_samples = int(len(ids) * sample_ratio)
    sampled_ids = random.sample(ids, num_samples)


    final_predictions = dict()
    actuals = dict()
    for prompt in predictions:
        final_predictions[prompt] = list()
        actuals[prompt] = list()
        for prediction in predictions[prompt]:
            if prediction['qas_id'] in sampled_ids:
                final_predictions[prompt].append(prediction['predicted_answer'])
                actuals[prompt].append(prediction['answer'])
        
    for prompt in predictions:
        print("Calculating BLEU score")
        bleu_score_out = bleu_score(final_predictions[prompt], actuals[prompt])
        print("Calculating ROUGE score")
        rouge_out = rouge_score(final_predictions[prompt], actuals[prompt])
        print("Calculating BLEU score")
        ppl_out = mean_perplexity(final_predictions[prompt])
        mcc_out = medical_concept_coverage(final_predictions[prompt], actuals[prompt], 8, device)
        print("Prompt: ", prompt)
        print("bleu:{0}, rouge:{1}, ppl:{2}, mcc:{3}, total_samples:{4}".format(bleu_score_out, rouge_out, ppl_out, mcc_out, len(final_predictions[prompt])))

def evaluate_predictions(prediction_file:str):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    random.seed(42)
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    # print(predictions)
    # # predictions_to_score = dict()
    # prompts = list(predictions.keys())

    # for prompt in predictions:
    # num_samples = int(len(predictions[prompts[0]]) * sample_ratio)
    # indices  = random.sample(range(0,len(predictions[prompts[0]])), num_samples)
    
    # for prompt in predictions:
    #     print(f"Prompt:{prompt}, total input: {len(predictions[prompt])}")

    # for prompt in predictions:
    #     predictions_to_score[prompt] = list()
    #     for prediction in predictions:

    #     predictions_to_score[prompt] = list(itemgetter(*indices)(predictions[prompt]))
    
    # with open('/scratch/rm6416/healthcare/results/predictions_to_score .json', 'w') as f:
    #     json.dump(predictions_to_score,f)

    final_predictions = dict()
    actuals = dict()
    for prompt in predictions:
        final_predictions[prompt] = list()
        actuals[prompt] = list()
        for prediction in predictions[prompt][:1]:
            final_predictions[prompt].append(prediction['predicted_answer'])
            actuals[prompt].append(prediction['answer'])
            # ids[].append(predictions["qas_id"])
    for prompt in predictions:
        print("Calculating BLEU score")
        bleu_score_out = metrics.bleu_score(final_predictions[prompt], actuals[prompt])
        print("Calculating ROUGE score")
        rouge_out = metrics.rouge_score(final_predictions[prompt], actuals[prompt])
        print("Calculating BLEU score")
        ppl_out = metrics.mean_perplexity(final_predictions[prompt])
        mcc_out = metrics.medical_concept_coverage(final_predictions[prompt], actuals[prompt], 8, device)
        print("Prompt: ", prompt)
        print("bleu:{0}, rouge:{1}, ppl:{2}, mcc:{3}, total_samples:{4}".format(bleu_score_out, rouge_out, ppl_out, mcc_out, len(final_predictions[prompt])))

evaluate_predictions('/home/rm6416/healthcare/results/predictions.json')
# sampled_evaluation('/home/rm6416/healthcare/results/predictions.json', 0.002)