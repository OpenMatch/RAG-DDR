import argparse
from rouge import Rouge
import json
import random
random.seed(42)
from tqdm import tqdm


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


def save_more_data(args, raw_data, dpo_data):
    match_list = ['raw', 'aug_1', 'aug_2','aug_3','aug_1-3']
    
    with open(args.out_path, 'w', encoding='utf-8') as f:
        for item in raw_data:
            dpo_item = find_item(item['id'],dpo_data)
            
            an = item['answers'][0]
            for sub_dpo_item in dpo_item['context']:
                rougel_score = _rougel_score(sub_dpo_item['text'], an)
                sub_dpo_item['rougel_score'] = rougel_score
                
            sorted_score = sorted_element_by_key(dpo_item['context'], 'rougel_score','type', match_list)
            
            max_score = sorted_score[0]
            if max_score['rougel_score'] !=0:
                scond_score = sorted_score[9]
            
                start = 9
                while (scond_score['text'] == max_score['text'] ):
                    if start == len(sorted_score):
                        break
                    scond_score = sorted_score[start]
                    start+=1
                    
                    
                inter_score = sorted_score[50]
                second_strat = 50
                
                while  (inter_score['text'] == max_score['text'] or inter_score['text'] == scond_score['text'] ) :
                    if second_strat == len(sorted_score):
                        break
                    inter_score = sorted_score[second_strat]
                    second_strat+=1
                    
                min_score = sorted_score[-1]
            
                if scond_score['text']!=min_score['text']:
                    item['chosen'] = max_score
                    item['rejected'] = scond_score
                    f.write(json.dumps(item) + '\n')
                    
                if inter_score['text']!=min_score['text']:
                    item['chosen'] = max_score
                    item['rejected'] = inter_score
                    f.write(json.dumps(item) + '\n')
                    
                item['chosen'] = max_score
                item['rejected'] = min_score
                f.write(json.dumps(item) + '\n')
                
                
            
            print("----------------")

            # max_score = find_max_element_by_key(dpo_item['context'], 'rougel_score','type', match_list)
            
            # for type in ['raw', 'aug_1', 'aug_2','aug_3','aug_1-3']:
                
            #     min_score = find_more_min_element_by_key(dpo_item['context'], 'rougel_score','type',type)                                     
            #     item['chosen'] = max_score
            #     item['rejected'] = min_score
                
            #     f.write(json.dumps(item) + '\n')

                # if max_score['rougel_score'] == 0.0 and min_score['rougel_score'] ==0.0:
                #     item['chosen'] = dpo_item[0]
                #     item['rejected'] = dpo_item[1]

            

def save_two_data(args, raw_data, dpo_data):
    with open(args.out_path, 'w', encoding='utf-8') as f:
        for item, dpo_item in zip(raw_data,dpo_data):
            
            dpo_item = [element for index, element in enumerate(dpo_item) if index != 2 and index != 3 and (index - 2) % 4 != 0 and (index - 3) % 4 != 0]
            an = item['answers'][0]
            for sub_dpo_item in dpo_item:
                rougel_score = _rougel_score(sub_dpo_item['text'], an)
                sub_dpo_item['rougel_score'] = rougel_score

            max_score = find_max_element_by_key(dpo_item, 'rougel_score')
            min_score = find_min_element_by_key(dpo_item, 'rougel_score')

            if max_score['rougel_score'] == 0.0 and min_score['rougel_score'] ==0.0:
                item['chosen'] = dpo_item[0]
                item['rejected'] = dpo_item[1]

            else:
                item['chosen'] = max_score
                item['rejected'] = min_score

            f.write(json.dumps(item) + '\n')
    
def save_radit_data(args, raw_data, dpo_data):
    with open(args.out_path, 'w', encoding='utf-8') as f:
        for item, dpo_item in tqdm(zip(raw_data,dpo_data)):
            
            # if item['data_type'] ==  'commonsense_qa':
                an = item['answer']
                new_dpo_item=[]
                for sub_dpo_item in dpo_item['context']:
                    if sub_dpo_item['type'] in ['raw','aug_1-5']:
                        if sub_dpo_item['data_type'] in ['math_qa', 'commonsense_qa']:
                            score = _acc_score(sub_dpo_item['text'][:2], an)
                            
                        if sub_dpo_item['data_type'] in ['wiki_qa','yahoo_answers_qa','marcoqa']:
                            score = _rougel_score(sub_dpo_item['text'], an)
                            
                        if sub_dpo_item['data_type'] in ['web_questions']:
                            score = _acc_score(sub_dpo_item['text'], an)
                            
                        if sub_dpo_item['data_type'] in ['gsm8k','strategyqa','aqua_rat','ecqa']: #minicpm: 'aqua_rat','ecqa'
                            cot_index = sub_dpo_item['text'].find("<COT")
                            # If "[COT]" is found, return the part before it, otherwise return the entire string
                            text = sub_dpo_item['text'][:cot_index] if cot_index != -1 else sub_dpo_item['text']
                            score = _acc_score(text, an)

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
                        default=None,
                        help="The path of original training/evaluation data file."
                        )
    parser.add_argument("--dpo_input_path", type=str, 
                        default=None,
                        help="The path of llm generated response file."
                        )
    parser.add_argument("--out_path", type=str, 
                        default=None,
                        help="Saved jsonl file path."
                        )

    args = parser.parse_args()
    raw_data = read_jsonl(args.raw_input_path)
    dpo_data = read_jsonl(args.dpo_input_path)
    
    #save_two_data(args, raw_data, dpo_data)
    # save_more_data(args, raw_data, dpo_data)
    save_radit_data(args, raw_data, dpo_data)

    print("---------------")

if __name__ == "__main__":
    main()
