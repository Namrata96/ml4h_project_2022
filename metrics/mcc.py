from transformers import AutoTokenizer, AutoModelForTokenClassification

from transformers import pipeline

def medical_concept_coverage(predictions, actuals, batch_size=1, device=-1):
    tokenizer = AutoTokenizer.from_pretrained("samrawal/bert-base-uncased_clinical-ner")

    model = AutoModelForTokenClassification.from_pretrained("samrawal/bert-base-uncased_clinical-ner")

    oracle = pipeline('token-classification', model=model, tokenizer=tokenizer, device=device, batch_size=batch_size, aggregation_strategy='first')
    # sentence = "However, the most common symptoms include: Nausea or recurrent upset stomach Abdominal bloating Abdominal pain Vomiting Indigestion Burning or gnawing feeling in the stomach between meals or at night Hiccups Loss of appetite Vomiting blood or coffee ground-like material Black, tarry stools To diagnose gastritis, your doctor will review your personal and family medical history, perform a thorough physical evaluation, and may recommend any of the following tests: Upper endoscopy. An endoscope, a thin tube containing a tiny camera, is inserted through your mouth and down into your stomach to look at the stomach lining. The doctor will check for inflammation and may perform a biopsy, a procedure in which a tiny sample of tissue is removed and then sent to a laboratory for analysis."
    
    predicted_out = oracle(predictions)
    actuals_out = oracle(actuals)
    predicted_out_set = set([d['word'] for d in predicted_out])
    actuals_out_set = set([d['word'] for d in actuals_out])
    common = predicted_out_set.intersection(actuals_out_set)
    if len(actuals_out_set) > 0:
        return len(common) / len(actuals_out_set)
    else:
        return 0