from transformers import AutoTokenizer, AutoModel
import json

model_path = 'PATH_TO_MODEL'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = model.eval()

def concat_messages(conversations, role, system):
    history = []
    first_query = system
    if conversations[0]['from'] == role:
        first_response = f"好的！现在我来扮演{role}。" + "我首先发话：" + conversations[0]['value']
    else:
        first_response = f"好的！现在我来扮演{role}。"
    
    history.append({"role": "user", "content": first_query})
    history.append({"role": "assistant", "content": first_response})
    
    for i in range(len(conversations)):
        if conversations[i]['from'] == role:
            if i ==0:
                continue
            else:
                assert conversations[i-1]['from'] != role
                query = f"{conversations[i-1]['from']}：" + conversations[i-1]['value']
                response = f"{conversations[i]['from']}：" + conversations[i]['value']
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
    assert conversations[-1]['from'] != role
    
    query = f"{conversations[-1]['from']}：" + conversations[-1]['value']
    return history, query

def make_inputs(context):
    dialogues= context.split('\n') 
    inputs = []  
    for dial in dialogues:
        role = dial.split("：")[0]
        dial = "：".join(dial.split("：")[1:])
        inputs.append({"from":role,"value":dial})
    return inputs

def get_response_chatglm(data):
    context = data['context']
    role = data['role']

    role_information = role_informations[role]
    role_system = f'''{role_information}
现在请你扮演一个角色扮演专家。请你根据上述信息扮演{role}进行对话。
''' 
    
    messages,query = concat_messages(make_inputs(context), role, role_system)
    response, _ = model.chat(tokenizer, query, messages)

    data["model_output"]=response

    return data


with open('data/test_data.jsonl','r') as f:
    datas = json.load(f)
with open('data/character_profiles.json','r') as f:
    role_informations = json.load(f)



results = []
for data in datas:
    results.append(get_response_chatglm(data))

f = open(f'results/generation.jsonl','w')  
f.write(json.dumps(results, ensure_ascii=False, indent=4))
f.flush()
