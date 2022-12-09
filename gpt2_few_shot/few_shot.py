from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
import torch
set_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("gpt2")

oracle = pipeline('question-answering', model='gpt2', device=device, batch_size=1)

anwer = oracle(question=question, context=context, max_answer_len=15, max_question_len=384, handle_impossible_answer=True)