



from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, set_seed
import json
import torch
import random
from tqdm.auto import tqdm
import os
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
os.environ['TRANSFORMERS_CACHE'] = '/scratch/rm6416/healthcare/new'
print(os.environ['TRANSFORMERS_CACHE'] )
# print(os.environ['XDG_CACHE_HOME'])

# print(os.environ['HF_HOME'])

# print("DEvice available: ", torch.cuda.current_device())
random.seed(42)
set_seed(42)

data_files = {'train': "/scratch/nm3571/healthcare/data/train.json",
'val': "/scratch/nm3571/healthcare/data/val.json",
'test': "/scratch/nm3571/healthcare/data/test.json"}
config_file = "/home/nm3571/healthcare/ml4h_project_2022/configs/prompts.json"
result_json = "/scratch/nm3571/healthcare/result/mashqa/predictions.json"

def run_gpt2(dataset_name: str):
    with open(data_file['train'], 'r') as f:
        train_data = json.load(f)
    with open(data_file['val'], 'r') as f:
        val_data = json.load(f)
    with open(data_file['test'], 'r') as f:
        test_data = json.load(f)
        test_data.extend(val_data)

    with open(config_file, 'r') as f:
        config = json.load(f)
    # configuration = GPT2Config()
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # model = AutoModelForCausalLM.from_config(configuration, force_download=True, cache_dir="/scratch/rm6416/healthcare/new/")
    # model = model.to(device)
    oracle = pipeline('text-generation', model='gpt2',tokenizer=tokenizer, device=device, batch_size=1)
    prompts = config['prompts'][dataset_name]
    answer_dict = {}
    print("Total test data:", len(test_data))
    count =0
    for prompt in prompts:
        print("Processing prompt: ", prompt)
        answer_dict[prompt] = list()
        # data = MyDataset(test_data[:1], prompt, prompts[prompt], train_data)
        for data in tqdm(test_data):
            question = data['question']
            if prompt == "zero_shot_empty_plus_grounding":
                context = data['context']
                input_text = prompts[prompt].format(context=context, question=question)
            elif prompt=="one_shot_plus_grounding":
                train_data_sample = random.sample(train_data,1)[0]
                # print(train_data_sample)
                context = data['context']
                shot_question = train_data_sample['question']
                shot_context = train_data_sample['context']
                shot_answer = train_data_sample['answer']
                input_text = prompts[prompt].format(shot_context=shot_context, shot_question=shot_question, shot_answer=shot_answer, context=context, question=question)
            else:#/ "zero_shot_empty", "zero_shot_cot"
                # print(prompts[prompt])
                input_text = prompts[prompt].format(question=question)
            # oracle.tokenizer.pad_token_id = oracle.model.config.eos_token_id
            # for answer in tqdm(oracle(data, prefix="the answer is", return_text=True, clean_up_tokenization_spaces=True, max_length=300, batch_size=8), total=len(data)):
            # print(answer)
            if len(input_text) < 1024:
              answer = oracle('', prefix=input_text, return_text=True, clean_up_tokenization_spaces=True, max_length=300)
              new_data = data
              new_data['predicted_answer'] = answer[0]['generated_text']
              answer_dict[prompt].append(new_data)

              # # print(data)
              # answer_dict[prompt].append(answer[0]['generated_text'])
              
            else:
              # print("More")
              count = count + 1
              continue
    print("Total not valid inputs", count)
    with open(result_json, 'w') as f:
        json.dump(answer_dict,f)
    print("Done")

run_gpt2('MASHQA')