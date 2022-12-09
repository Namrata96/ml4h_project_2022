from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
set_seed(42)



tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = AutoModelForCausalLM.from_pretrained("gpt2")

generator = pipeline('text-generation', model='gpt2')

generator("Hello, I'm a language model,", max_length=10, num_return_sequences=5)