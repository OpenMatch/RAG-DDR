import json
import argparse
import csv
from tqdm import tqdm
import pandas as pd

def load_jsonl(data_path):
    all_data = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            data = json.loads(line)
            all_data.append(data)
    return all_data
            
def read_csv_to_list(filepath, delimiter=' '):
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        data_list = [row for row in reader] 
    return data_list
def chunk_list(original_list, chunk_size):
    return [original_list[i:i + chunk_size] for i in range(0, len(original_list), chunk_size)]

def load_psg_from_wiki(id,corpus):
    psg = corpus[id]
    return psg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", type=str, 
                        default=None)
    parser.add_argument("--trec_save_path", type=str, nargs='+',
                        default= None)
    parser.add_argument("--input_path", type=str, nargs='+',
                        default= None)
    parser.add_argument("--topk", type=int, 
                        default=100)
    parser.add_argument("--selectk", type=int, 
                        default=100)
    parser.add_argument("--output_path", type=str, 
                        default=None)
    args = parser.parse_args()
    
    wiki_text = load_jsonl(args.corpus_path)
    all_trec_list = args.trec_save_path
    all_input_list = args.input_path
    all_output_list = args.output_path

    for i in range(len(all_input_list)):
        trec_list = read_csv_to_list(all_trec_list[i])
        sub_trec=chunk_list(trec_list,args.topk)
        task_input = load_jsonl(all_input_list[i])
        
        for idx,item in enumerate(task_input):
            trec_item = sub_trec[idx][:args.selectk]
            task_input[idx]['passage']=[]
            for psg in trec_item:    
                    psg_id = int(psg[2])
                    passage = load_psg_from_wiki(psg_id ,wiki_text)
                    task_input[idx]['passage'].append(passage)
        
        with open(all_output_list[i], 'w') as file:
            for item in task_input:
                json_line = json.dumps(item) 
                file.write(json_line + '\n')  
    
if __name__ == "__main__":
    main()