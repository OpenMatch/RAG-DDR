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
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:
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

def get_topn_score(dpo_item, item, top_n):
    new_dpo_item = dpo_item[:top_n]
    return new_dpo_item

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
            an = item['answer']
            new_dpo_item=[]
            data_type=item['data_type']
            for id, sub_dpo_item in enumerate(dpo_item['context']):
            
                if args.llama_style:
                    if data_type in ['math_qa', 'commonsense_qa','aqua_rat']:
                        score = _acc_score(sub_dpo_item['text'][:3], an)

                    if data_type in ['ecqa']:
                        score = _acc_score(sub_dpo_item['text'], an)

                    if data_type in ['gsm8k','strategyqa']: #minicpm: 'aqua_rat','ecqa'
                        cot_index = sub_dpo_item['text'].find("COT")
                        # If "[COT]" is found, return the part before it, otherwise return the entire string
                        text = sub_dpo_item['text'][:cot_index] if cot_index != -1 else sub_dpo_item['text']
                        score = _acc_score(text, an)

                else:
                    if data_type in ['math_qa', 'commonsense_qa']:
                        score = _acc_score(sub_dpo_item['text'][:3], an)

                    if data_type in ['gsm8k','strategyqa','aqua_rat','ecqa']:
                        cot_index = sub_dpo_item['text'].find("COT")
                        # If "[COT]" is found, return the part before it, otherwise return the entire string
                        text = sub_dpo_item['text'][:cot_index] if cot_index != -1 else sub_dpo_item['text']
                        score = _acc_score(text, an)

                if data_type in ['wiki_qa','yahoo_answers_qa','marcoqa']:
                        score = _rougel_score(sub_dpo_item['text'], an)
                        
                if data_type in ['web_questions']:
                        score = _acc_score(sub_dpo_item['text'], an)
                    
                    
                sub_dpo_item['score'] = score
                sub_dpo_item['id'] =id
                new_dpo_item.append(sub_dpo_item)

            new_dpo_item =get_topn_score(new_dpo_item, item, args.top_n)

            max_list = find_max_element_by_key(new_dpo_item, 'score')
            min_list = find_min_element_by_key(new_dpo_item, 'score')

            if max_list == min_list:
                continue

            max_passage, min_passage = find_min_max_passage(max_list,min_list,item,args.top_n)

            for psg in [max_passage, min_passage]:
                if psg==[]:
                    continue

                item['error_passage'] = psg

                if psg == max_passage:
                    item['chosen'] = 'YES'
                    item['rejected'] = 'NO'
                else:
                    item['chosen'] = 'NO'
                    item['rejected'] = 'YES'

                f.write(json.dumps(item) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_input_path", type=str, 
                        default=None,
                        help="The path of original training/evaluation data file.")
    
    parser.add_argument("--dpo_input_path", type=str, 
                        default=None,
                        help="The path of responses that llm generate.")
    
    parser.add_argument("--out_path", type=str, 
                        default=None,
                        help="Saved jsonl file path.")
    
    parser.add_argument("--llama_style", action='store_true',
                        default=None,
                        help="Is the DPO data generated by llama.")
    
    parser.add_argument("--top_n", type=int, 
                        default=100,
                        help="n passage need to refine.")

    args = parser.parse_args()
    raw_data = read_jsonl(args.raw_input_path)
    dpo_data = read_jsonl(args.dpo_input_path)
    
   
    save_radit_data(args, raw_data, dpo_data)

    print("-----finish----------")

if __name__ == "__main__":
    main()
