
import json
import random
from operator import itemgetter 

import evaluate
from transformers import AutoTokenizer, AutoModelForTokenClassification

from transformers import pipeline
from evaluate import load
import torch
import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/rm6416/healthcare/new'
print(os.environ['TRANSFORMERS_CACHE'] )

def bleu_score(predictions, actuals):
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=actuals)
    return results['bleu']

def medical_concept_coverage(predictions, actuals, batch_size=1, device=-1):
    tokenizer = AutoTokenizer.from_pretrained("samrawal/bert-base-uncased_clinical-ner")

    model = AutoModelForTokenClassification.from_pretrained("samrawal/bert-base-uncased_clinical-ner")

    oracle = pipeline('token-classification', model=model, tokenizer=tokenizer, device=device, batch_size=batch_size, aggregation_strategy='first')
    # sentence = "However, the most common symptoms include: Nausea or recurrent upset stomach Abdominal bloating Abdominal pain Vomiting Indigestion Burning or gnawing feeling in the stomach between meals or at night Hiccups Loss of appetite Vomiting blood or coffee ground-like material Black, tarry stools To diagnose gastritis, your doctor will review your personal and family medical history, perform a thorough physical evaluation, and may recommend any of the following tests: Upper endoscopy. An endoscope, a thin tube containing a tiny camera, is inserted through your mouth and down into your stomach to look at the stomach lining. The doctor will check for inflammation and may perform a biopsy, a procedure in which a tiny sample of tissue is removed and then sent to a laboratory for analysis."
    predictions_filtered = list()
    actual_filtered = list()
    for actual, prediction in zip(predictions, actuals):
        prediction = prediction.encode("ascii", "ignore")
        actual = actual.encode("ascii", "ignore")
        if actual != None and actual != "" and prediction != None and prediction != "":
            predictions_filtered.append(str(prediction))
            actual_filtered.append(str(actual))
    print("Samples after filtering", len(predictions_filtered))
    # print(type(predictions_filtered))
    # print(predictions_filtered[0])
    print(type(actual_filtered))
    print(actual_filtered[0])
    predicted_out = oracle(predictions_filtered)
    actuals_out = oracle(actual_filtered)
    # print(predicted_out)
    predicted_out_set = set([l['word'] for d in predicted_out for l in d])
    actuals_out_set = set([l['word'] for d in actuals_out for l in d])
    common = predicted_out_set.intersection(actuals_out_set)
    if len(actuals_out_set) > 0:
        return len(common) / len(actuals_out_set)
    else:
        return 0


def mean_perplexity(predictions):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=predictions, model_id='gpt2')
    return round(results['mean_perplexity'],3)


def rouge_score(predictions, actuals):
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=actuals)
    return results # Returns: {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}

def sampled_evaluation(prediction_file:str, sample_ratio:int):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Running sampled evaluation")
    random.seed(42)
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)

    ids = dict()
    for prompt in predictions:
        ids[prompt] = set()
        for prediction in predictions[prompt]:
            ids[prompt].add(prediction['qas_id'])
    
    initial_prompts = list(predictions.keys())
    
    existing_ids = ids[initial_prompts[0]]
    setlist = list()
    for prompt in predictions:
        print("prompt", prompt)
        print("Length of ids", len(ids[prompt]))
        setlist.append(ids[prompt])

    interset_ids = existing_ids.intersection(*setlist)
    # existing_ids = list(existing_ids)
    print("Number of intersecting ids: ", len(interset_ids))

    num_samples = int(len(interset_ids) * sample_ratio)
    sampled_ids = random.sample(list(interset_ids), num_samples)
    with open('/scratch/rm6416/healthcare/result/mashqa_intersect_ids.json', 'w') as f:
        json.dump(sampled_ids, f)

    final_predictions = dict()
    actuals = dict()
    sampled_preds = dict()
    for prompt in predictions:
        final_predictions[prompt] = list()
        actuals[prompt] = list()
        sampled_preds[prompt] = list()
        for prediction in predictions[prompt]:
            if prediction['qas_id'] in sampled_ids:
                sampled_preds[prompt].append(prediction)
                final_predictions[prompt].append(prediction['predicted_answer'])
                actuals[prompt].append(prediction['answer'])
    # with open('/scratch/rm6416/healthcare/result/mashqa_sampled_answers.json', 'w') as f:
    #     json.dump(sampled_preds, f)

    for prompt in predictions:
        print(len(final_predictions[prompt]))
        print(len(actuals[prompt]))
        # print("Calculating BLEU score")
        # bleu_score_out = bleu_score(final_predictions[prompt], actuals[prompt])
        # print("Calculating ROUGE score")
        # rouge_out = rouge_score(final_predictions[prompt], actuals[prompt])
        # print("Calculating PPL score")
        # ppl_out = mean_perplexity(final_predictions[prompt])
        print("Calculating MCC score")
        mcc_out = medical_concept_coverage(final_predictions[prompt], actuals[prompt], 8, device)
        print("Prompt: ", prompt)
        print(f"mcc:{mcc_out}, total_samples:{len(final_predictions[prompt])}")
        # print(f"bleu:{bleu_score_out}, rouge:{rouge_out}, ppl:{ppl_out}, total_samples:{len(final_predictions[prompt])}")


# with open('/home/rm6416/healthcare/results/predictions.json', 'r') as f:
#     predictions = json.load(f)
# with open('/scratch/rm6416/healthcare/result/predictions_one_shot.json', 'r') as f:
#     one_shot_pred = json.load(f)
# with open('/scratch/rm6416/healthcare/result/predictions_zero_shot_grounding.json', 'r') as f:
#     zero_shot_ground = json.load(f)
# predictions['one_shot_plus_grounding'] = one_shot_pred['one_shot_plus_grounding']
# predictions['zero_shot_empty_plus_grounding'] = zero_shot_ground['zero_shot_empty_plus_grounding']

# with open('/scratch/rm6416/healthcare/result/all_predictions.json', 'w') as f:
#     json.dump(predictions, f)

# evaluate_predictions('/home/rm6416/healthcare/results/predictions.json')
sampled_evaluation('/scratch/rm6416/healthcare/result/all_predictions.json', 0.09)

# final b,r,p fot gpt 2 = out_28196637
# final mcc for gpt2 = 28199160