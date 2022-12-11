import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(50)

class GPT2GenQADataset(Dataset):

    def __init__(self, input_txt, answer_txt, tokenizer, gpt2_type = "gpt2", max_length = 768):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.answer_ids = []
#         self.input_attn_masks = []
#         self.answer_attn_masks = []
        
        assert len(input_txt) == len(answer_txt)

        for input_txt, answer_txt in zip(input_txt, answer_txt):
            
            input_encodings_dict = tokenizer(input_txt, truncation=True, max_length = max_length, padding="max_length")
            
            answer_encodings_dict = tokenizer(answer_txt, truncation = True, max_length = max_length, padding = "max_length")
            
            self.input_ids.append(torch.tensor(input_encodings_dict['input_ids']))
#             self.input_attn_masks.append(torch.tensor(input_encodings_dict['attention_mask']))
            self.answer_ids.append(torch.tensor(answer_encodings_dict['input_ids']))
#             self.answer_attn_masks.append(torch.tensor(answer_encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx] , self.answer_ids[idx]