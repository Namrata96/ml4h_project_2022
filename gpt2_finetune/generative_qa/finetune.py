import pandas as pd
import json
import numpy as np
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from genQA_dataset import GPT2GenQADataset

seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

device = torch.device("cuda")

root_dir = "/scratch/as14770/ML4H/ml4h_project_2022"
train_csv = 'Data/MedMCQA/train_for_abstractive_qa.csv'

train_data = pd.read_csv(os.path.join(root_dir, train_csv))

train_data.dropna(inplace = True)

print("Num of question answers pairs: ", len(train_data))

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token = '<|pad|>')

model = GPT2LMHeadModel.from_pretrained('gpt2')

train_dataset = GPT2GenQADataset(train_data['input_text'], train_data['text'], tokenizer)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
batch_size = 10
train_split, val_split = random_split(train_dataset, [train_size, val_size])

train_dataloader = DataLoader(
            train_split,  # The training samples.
            sampler = RandomSampler(train_split), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )


validation_dataloader = DataLoader(
            val_split, # The validation samples.
            sampler = SequentialSampler(val_split), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )


configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)


model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

epochs = 25
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

patience = 2

optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon)

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

train_stats = {'train_step_loss' : [], 'epoch_train_loss': [], 'epoch_val_loss': []}
best_val_loss = 1000
best_epoch_idx = 0

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    for step, batch in enumerate(train_dataloader):
        input_ids = batch[0].to(device)
        answer_ids = batch[1].to(device)
        attn_masks = batch[2].to(device)
        model.zero_grad()
        outputs = model(input_ids, labels = answer_ids, attention_mask = attn_masks)
        loss = outputs[0]  
        batch_loss = loss.item()
        print("Step Loss: ", batch_loss)
        train_stats['train_step_loss'].append(batch_loss)
        epoch_train_loss += batch_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
    epoch_train_loss = epoch_train_loss / len(train_dataloader)
    train_stats['epoch_train_loss'].append(epoch_train_loss)
    print("Train Loss: ", epoch_train_loss)
    
    model.eval()
    epoch_val_loss = 0
    for val_step, val_batch in enumerate(validation_dataloader):
        input_ids = val_batch[0].to(device)
        answer_ids = val_batch[1].to(device)
        attn_masks = val_batch[2].to(device)
        with torch.no_grad():
            outputs  = model(input_ids, labels = answer_ids, attention_mask = attn_masks)
            loss = outputs[0]  
        epoch_val_loss += loss.item()
        
    epoch_val_loss = epoch_val_loss / len(validation_dataloader)
    train_stats['epoch_val_loss'].append(epoch_val_loss)
    print("Val Loss: ", epoch_val_loss)
    
    if epoch == 0:
        best_val_loss = epoch_val_loss
    
    if epoch_val_loss <= best_val_loss:
        torch.save(model.state_dict(), 'gen_qa_gpt2_finetuned_'+str(epoch)+'.pth')
        best_val_loss = epoch_val_loss
        best_epoch_idx = epoch
    
    if epoch - best_epoch_idx >= patience:
        break
        
print("Finetuning complete!")

train_stats_df = pd.DataFrame(train_stats)
train_stats_df.to_csv("gen_qa_gpt2_finetuning.csv")