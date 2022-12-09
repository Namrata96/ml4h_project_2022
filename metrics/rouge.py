import evaluate

def rouge_score(predictions, actuals)
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=actuals)
    return results # Returns: {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0, 'rougeLsum': 1.0}
