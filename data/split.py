import json
import random

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

data_path = '/home/lixz23/ragsft/data/old_marco/train_v2.1_marco2.1_psg.jsonl'
output_train_path = '/data/groups/QY_LLM_Other/meisen/dataset/marco2.1/train.jsonl'
output_dev_path = '/data/groups/QY_LLM_Other/meisen/dataset/marco2.1/dev.jsonl'

# 设置随机种子
random.seed(42)

# 读取数据集
data = read_jsonl(data_path)

# 随机打乱数据
random.shuffle(data)

# 划分数据
dev_data = data[:3000]
train_data = data[3000:]

# 将数据写入文件
write_jsonl(train_data, output_train_path)
write_jsonl(dev_data, output_dev_path)

print('finish!')