import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data
data = read_jsonl('/data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval/hotpotqa/dev_psg.jsonl')

print("-------")