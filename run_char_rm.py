import sys
import torch
import json
from BaichuanCharRM.modeling_baichuan import BaichuanCharRM
from BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer

max_seq_length = 4096

with open("data/character_profiles.json", "r") as f:
    character_profile = json.load(f)
with open(f"results/generation_trans_baichuan_7b.jsonl", mode='r') as f:
    records = json.load(f)
reward_model_path = 'BaichuanCharRM/'

def format_input(example):
    input_text = "<RoleInfo>\n\n" \
        + str(character_profile[example['role']]) + "\n\n<Context>\n\n" + example['context'] + "\n\n<Response>\n\n" + example['model_output'] + "\n\n<Dimension>\n\n" + example["metric_zh"]
    return input_text

tokenizer = BaichuanTokenizer.from_pretrained(reward_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" 
base_model = BaichuanCharRM.from_pretrained(reward_model_path, torch_dtype=torch.bfloat16).cuda()


import tqdm

for record in tqdm.tqdm(records):
    input_text = format_input(record)
    input_ids = tokenizer.encode(text=input_text, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
    with torch.no_grad():
        score = base_model(input_ids=input_ids)[1].item() * 4 + 1
        record[record['metric_en']] = score

f = open('results/evaluation.jsonl','w')  
f.write(json.dumps(records, ensure_ascii=False, indent=4))
