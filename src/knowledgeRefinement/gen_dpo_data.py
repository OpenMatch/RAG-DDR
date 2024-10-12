import argparse
import re
from rouge import Rouge
import json
import random
import itertools
from tqdm import tqdm
random.seed(42)

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def _acc_score(prediction, ground_truth):
    if ground_truth in prediction or ground_truth.lower() in prediction or ground_truth.capitalize() in prediction:
        return 1.0
    else:
        return 0.0

def read_jsonl(file_path):

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def find_min_element_by_key(data_list, key):
    min_element = min(data_list, key=lambda x: x[key])
    min__list = [element for element in data_list if element[key]== min_element[key]]
    return min__list

def find_max_element_by_key(data_list, key):
    max_element = max(data_list, key=lambda x: x[key])
    max__list = [element for element in data_list if element[key] == max_element[key]]
    return max__list

# def find_max_element_by_key(data_list, key,match_key, match_type_list):    
#     filtered_elements = [d for d in data_list if d.get(match_key) in  match_type_list]
#     return max(filtered_elements, key=lambda x: x[key])

def find_more_min_element_by_key(data_list, key, match_key, match_type):
    filtered_elements = [d for d in data_list if d.get(match_key) == match_type]
    return min(filtered_elements, key=lambda x: x[key])

def sorted_element_by_key(data_list, key,match_key, match_type_list):
    filtered_elements = [d for d in data_list if d.get(match_key) in  match_type_list]
    return sorted(filtered_elements, key=lambda x: x[key], reverse=True)
    

def find_item(id,dpo_data):
    for item in dpo_data:
        if id == item['id']:
            return item
    return None

def extract_index(s):
    match = re.search(r'aug_(\d+)', s)
    if match:
        return int(match.group(1))
    else:
        return None

def find_extremes(reordered, target):
    # 找到目标数组中的每个元素在重排序数组中的索引
    indices = [reordered.index(x) for x in target]
    
    # 找到最小和最大的索引
    min_index = min(indices)
    max_index = max(indices)
    
    # 根据最小和最大索引返回元素
    return reordered[min_index], reordered[max_index]



def get_topn_score(dpo_item, item, top_n):
    new_dpo_item = dpo_item[:top_n]
    return new_dpo_item

def get_max_min_score_random_list(max_list, min_list):
    max_index_list = []
    for max_element in max_list:
        index = extract_index(max_element['type'])
        max_index_list.append(index)

    min_index_list = []  
    for min_element in min_list:
        index = extract_index(min_element['type'])
        min_index_list.append(index)

    return max_index_list, min_index_list

def get_true_false_different(max_list,min_list,new_judges,item,top_n):
    passage = item['passage'][:top_n]
    
    False_error_passage = []
    True_error_passage = []

    for max in max_list:
        if new_judges[max['id']] == False:
            False_error_passage.append(passage[max['id']])

    for min in min_list:
        if new_judges[min['id']] == True:
            True_error_passage.append(passage[min['id']])

    return False_error_passage, True_error_passage


def find_min_max_passage(max_list,min_list,item,top_n):
    passage = item['passage'][:top_n]

    max_passage = []
    min_passage = []

    for max in max_list:
        max_passage.append(passage[max['id']])

    for min in min_list:
        min_passage.append(passage[min['id']])

    return max_passage, min_passage

def save_radit_data(args, raw_data, dpo_data):
    with open(args.out_path, 'w', encoding='utf-8') as f:
        for item, dpo_item in tqdm(zip(raw_data,dpo_data)):
            
            # if item['data_type'] ==  'commonsense_qa':
                an = item['answer']
                new_dpo_item=[]
                for id, sub_dpo_item in enumerate(dpo_item['context']):
                
                    if sub_dpo_item['data_type'] in ['math_qa', 'commonsense_qa','aqua_rat','ecqa']:
                        score = _acc_score(sub_dpo_item['text'][:2], an)
                        
                    if sub_dpo_item['data_type'] in ['wiki_qa','yahoo_answers_qa','marcoqa']:
                        score = _rougel_score(sub_dpo_item['text'], an)
                        
                    if sub_dpo_item['data_type'] in ['web_questions']:
                        score = _acc_score(sub_dpo_item['text'], an)
                        
                    if sub_dpo_item['data_type'] in ['gsm8k','strategyqa']: #minicpm: 'aqua_rat','ecqa'
                        cot_index = sub_dpo_item['text'].find("COT")
                        # 如果找到 "[COT]"，返回它之前的部分，否则返回整个字符串
                        text = sub_dpo_item['text'][:cot_index] if cot_index != -1 else sub_dpo_item['text']
                        score = _acc_score(text, an)
                        
                        
                    sub_dpo_item['score'] = score
                    sub_dpo_item['id'] =id
                    new_dpo_item.append(sub_dpo_item)

                new_dpo_item =get_topn_score(new_dpo_item, item, args.top_n)#选择rerank后的top_n去用于dpo训练
                # new_judges = get_topn_score(item['judge_preds'], item, args.top_n)

                max_list = find_max_element_by_key(new_dpo_item, 'score')
                min_list = find_min_element_by_key(new_dpo_item, 'score')

                if max_list == min_list:
                    continue

                max_passage, min_passage = find_min_max_passage(max_list,min_list,item,args.top_n)


                #False_error_passage, True_error_passage = get_true_false_different(max_list,min_list,new_judges,item, args.top_n)


                # for error in [False_error_passage, True_error_passage]:

                #     if error==[]:
                #         continue

                #     item['error_passage'] = error

                #     if error  == False_error_passage:
                #         item['chosen'] = 'YES'
                #         item['rejected'] = 'NO'
                #     else:
                #         item['chosen'] = 'NO'
                #         item['rejected'] = 'YES'

                #     f.write(json.dumps(item) + '\n')

                for ee in [max_passage, min_passage]:
                    if ee==[]:
                        continue

                    item['error_passage'] = ee

                    if ee  == max_passage:
                        item['chosen'] = 'YES'
                        item['rejected'] = 'NO'
                    else:
                        item['chosen'] = 'NO'
                        item['rejected'] = 'YES'

                    f.write(json.dumps(item) + '\n')




            
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_input_path", type=str, default="/data/home/lixz23/ragsft/data/marco_v2.1/bge_large_retriever_128_256_top100/retriever_train_4000_noread_psg.jsonl")
    parser.add_argument("--dpo_input_path", type=str, default="/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/double_train_rerank/double_llama_2300_reward_random_100/train_list/all_text_data_psg.jsonl")
    parser.add_argument("--out_path", type=str, default="/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/double_train_rerank/double_llama_2300_reward_random_100/train.jsonl")
    parser.add_argument("--top_n", type=int, default=100)

    args = parser.parse_args()
    raw_data = read_jsonl(args.raw_input_path)
    dpo_data = read_jsonl(args.dpo_input_path)
    
   
    save_radit_data(args, raw_data, dpo_data)

    print("-----finish----------")

if __name__ == "__main__":
    main()
