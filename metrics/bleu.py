import evaluate

def bleu_score(predictions, actuals)
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=actuals)
    return results['bleu']