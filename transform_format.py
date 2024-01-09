import json
import copy


with open('data/id2metric.jsonl','r') as f:
    id_metric = json.load(f)

with open(f'results/generation_trans_baichuan_7b.jsonl','r') as f:
    datas = json.load(f)

results = []

for data in datas:
    if data['model_output'] != "ERROR":
        model_output = data['model_output'].split("\n")[0] # Prevent continuous generation
        data['model_output'] = model_output
        for x in id_metric[str(data['id'])]:
            data['metric_en']= x[0]
            data['metric_zh']= x[1]
            tmp = copy.deepcopy(data)
            results.append(tmp)

f= open(f"results/generation_trans.jsonl", 'w') 
f.write(json.dumps(results, ensure_ascii=False, indent=4)) 
f.close()