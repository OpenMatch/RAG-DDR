import argparse
from rouge import Rouge
import json
import random
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


def get_max_min_score(max_list,min_list):
    max_score = None
    for max_element in max_list:
        if max_element['type'] == 'raw':
            max_score = max_element
            break
                
    if max_score == None:
        max_score = max_list[0]
        
    min_score = None             
    if max_score['type'] == 'raw':
        for min_element in min_list:
            if min_element['type']!='raw':
                min_score = min_element
                break
            
    if min_score == None:
        min_score = min_list[0]
        
    return max_score, min_score

                
def save_radit_data(args, raw_data, dpo_data):

    with open(args.out_path, 'w', encoding='utf-8') as f:
        for item, dpo_item in tqdm(zip(raw_data,dpo_data)):
            an = item['answer']
            new_dpo_item=[]

            for sub_dpo_item in dpo_item['context']:
                if sub_dpo_item['type'] in ['raw','aug_1-5']:
                    
                    if args.llama_style:
                        if sub_dpo_item['data_type'] in ['math_qa', 'commonsense_qa','aqua_rat']:
                           score = _acc_score(sub_dpo_item['text'][:3], an)

                        if sub_dpo_item['data_type'] in ['ecqa']:
                            score = _acc_score(sub_dpo_item['text'], an)

                        if sub_dpo_item['data_type'] in ['gsm8k','strategyqa']: #minicpm: 'aqua_rat','ecqa'
                            cot_index = sub_dpo_item['text'].find("COT")
                            # If "[COT]" is found, return the part before it, otherwise return the entire string
                            text = sub_dpo_item['text'][:cot_index] if cot_index != -1 else sub_dpo_item['text']
                            score = _acc_score(text, an)

                    else:
                        if sub_dpo_item['data_type'] in ['math_qa', 'commonsense_qa']:
                            score = _acc_score(sub_dpo_item['text'][:3], an)

                        if sub_dpo_item['data_type'] in ['gsm8k','strategyqa','aqua_rat','ecqa']:
                            cot_index = sub_dpo_item['text'].find("COT")
                            # If "[COT]" is found, return the part before it, otherwise return the entire string
                            text = sub_dpo_item['text'][:cot_index] if cot_index != -1 else sub_dpo_item['text']
                            score = _acc_score(text, an)
                        
                    if sub_dpo_item['data_type'] in ['wiki_qa','yahoo_answers_qa','marcoqa']:
                        score = _rougel_score(sub_dpo_item['text'], an)
                        
                    if sub_dpo_item['data_type'] in ['web_questions']:
                        score = _acc_score(sub_dpo_item['text'], an)

                    sub_dpo_item['score'] = score
                    new_dpo_item.append(sub_dpo_item)
                    
            max_list = find_max_element_by_key(new_dpo_item, 'score')
            min_list = find_min_element_by_key(new_dpo_item, 'score')

            random.shuffle(max_list)
            random.shuffle(min_list)

            
            max_score, min_score = get_max_min_score(max_list,min_list)

            if max_score['score'] == min_score['score']:
                continue
            item['chosen'] = max_score
            item['rejected'] = min_score
                
            f.write(json.dumps(item) + '\n')

            
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_input_path", type=str, 
                        default='/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/dev_noread/dev_psg.jsonl',
                        help="The path of original training/evaluation data file."
                        )
    parser.add_argument("--dpo_input_path", type=str, 
                        default='/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/train_LLM_data/llama3-8b_top5/marco_LLM_rerank/dev_list/all_text.jsonl',
                        help="The path of responses that llm generate."
                        )
    parser.add_argument("--out_path", type=str, 
                        default='/home/lixz23/ragsft/RAG-DDR/data/test_data/train_llama/dev.jsonl',
                        help="Saved jsonl file path."
                        )
    
    parser.add_argument("--llama_style", action='store_true',
                        default=True,
                        help="Is the DPO data generated by llama."
                        )

    args = parser.parse_args()
    raw_data = read_jsonl(args.raw_input_path)
    dpo_data = read_jsonl(args.dpo_input_path)

    save_radit_data(args, raw_data, dpo_data)

    print("---------------")

if __name__ == "__main__":
    main()
