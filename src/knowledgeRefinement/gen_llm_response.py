from vllm import LLM, SamplingParams
import argparse
import json
from tqdm import  tqdm
from rouge import Rouge
import torch
from transformers import AutoTokenizer, DefaultDataCollator
from torch.utils.data import Dataset, DataLoader, SequentialSampler,DistributedSampler
from dataclasses import dataclass, field
from transformers import AutoTokenizer
import os
from template import minicpm_prompt_template,augment_templeta,multi_choice,QA_templeta,Mult_COT_templeta,QA_COT_templeta,COT_few_shot,COT_question,\
aqua_rat_shot,ecqa_shot,gsm8k_shot,strategyqa_shot,llama_multi_choice,llama_QA_COT_templeta
SPECIAL_TOKEN_LENGTH=64

def read_jsonl(file_path):

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            item['id'] = idx
            data.append(item)
    return data

class llmDataset(Dataset):
    def __init__(self,data_list,args,tokenizer):
        self.data_list = data_list
        self.args = args
        self.tokenizer = tokenizer
        
    def process_query_with_psg(self, item):

        if self.args.llama_style:
            if item['data_type'] in ['commonsense_qa', 'math_qa',"aqua_rat","ecqa"]:
                raw_input = llama_multi_choice.format(item['question'])

            if item['data_type'] == ['web_questions','wiki_qa','yahoo_answers_qa',"marcoqa"]:
                raw_input = QA_templeta.format(item['question'])
               
            if item['data_type'] in ["gsm8k","strategyqa"]:
                if item['data_type'] == "gsm8k":
                    shot_example = gsm8k_shot
                    
                if item['data_type'] =="strategyqa":
                    shot_example = strategyqa_shot
                    
                shot_prompt =""
                for few_shot in shot_example:
                    shot_prompt = shot_prompt+COT_few_shot.format(few_shot["question"],few_shot["answer"],few_shot["cot"])
                raw_input = llama_QA_COT_templeta + shot_prompt + COT_question.format(item['question'])

            item['raw_input'] = [ {"role": "user", "content": raw_input},]
            item['raw_input'] = self.tokenizer.apply_chat_template(item['raw_input'], add_generation_prompt=True, tokenize=False)

        else:
            if item['data_type'] in ['commonsense_qa','math_qa']:
                raw_input = multi_choice.format(item['question'])
            
            if item['data_type'] == ['web_questions','wiki_qa','yahoo_answers_qa',"marcoqa"]:
                raw_input = QA_templeta.format(item['question'])
                
            if item['data_type'] in ["aqua_rat","ecqa"]:
                
                if item['data_type'] == "aqua_rat":
                    shot_example = aqua_rat_shot
                
                if item['data_type'] =="ecqa":
                    shot_example = ecqa_shot
                                
                shot_prompt =""
                for few_shot in shot_example:
                    shot_prompt = shot_prompt+COT_few_shot.format(few_shot["question"],few_shot["answer"],few_shot["cot"])
                raw_input = Mult_COT_templeta + shot_prompt + COT_question.format(item['question']) 
                
            if item['data_type'] in ["gsm8k","strategyqa"]:
                
                if item['data_type'] == "gsm8k":
                    shot_example = gsm8k_shot
                    
                if item['data_type'] =="strategyqa":
                    shot_example = strategyqa_shot
                    
                shot_prompt =""
                for few_shot in shot_example:
                    shot_prompt = shot_prompt+COT_few_shot.format(few_shot["question"],few_shot["answer"],few_shot["cot"])
                raw_input = QA_COT_templeta + shot_prompt + COT_question.format(item['question']) 
            item['raw_input'] = minicpm_prompt_template.format(raw_input)
    
        item = self.get_more_psg(item,raw_input,)  
        return item


    
    def get_more_psg(self,item,raw_input):
        item['augment_input']=[]
        item['augment_type'] = []

        psg_list = []
        psgs = item['passage']
        for p in psgs:
             psg_list.append(p['segment'])
        token_query = self.tokenizer([raw_input])
        query_length = len(token_query.input_ids[0])                
      
        for idx, psg in enumerate(psg_list):

            token_aug_psg = self.tokenizer([psg])
            token_aug_psg = token_aug_psg.input_ids[0][:4096-SPECIAL_TOKEN_LENGTH-query_length]
            new_psg = self.tokenizer.decode(token_aug_psg,skip_special_tokens=True)
        
            if self.args.llama_style:
                aug_input = [{"role": "user", "content": augment_templeta.format(new_psg, raw_input)}]
                aug_input = self.tokenizer.apply_chat_template(aug_input, add_generation_prompt=True, tokenize=False)
                item['augment_input'].append(aug_input)
            
            else:
                item['augment_input'].append(minicpm_prompt_template.format(
                augment_templeta.format(new_psg, raw_input)))
                 
            item['augment_type'].append('aug_'+ str(idx))

        return item
            
    def __getitem__(self, index):
        item = self.data_list[index]       
        item = self.process_query_with_psg(item)
        return item
    
    def __len__(self):
        return len(self.data_list)
    
    def Collactor (self, batch):
        raw_input = [f['raw_input'] for f in batch]
        id = [f['id'] for f in batch]
        answers = [f['answer'] for f in batch]
        augment_input = [f['augment_input'] for f in batch]
        type = [f['augment_type'] for f in batch]
        data_type = [f['data_type'] for f in batch]
        
        return{'id':id,
               'raw_input':raw_input,
               'answers':answers,
               'augment_input':augment_input,
               'type': type,
               'data_type':data_type,
               }
        
def split_list_evenly(lst, n):
    length = len(lst)
    avg = length // n
    remainder = length % n
    out = []
    start = 0

    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        out.append(lst[start:end])
        start = end
    
    return out       
        
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        default="/home/lixz23/ragsft/data/marco_v2.1/bge_large_retriever_128_256_top100/retriever_train_4000_noread_psg.jsonl")
    parser.add_argument('--model_name_or_path', type=str,
                        default="/home/lixz23/pretrain-model/Llama3-8b-instruct")
    parser.add_argument('--output_path', type=str,
                        default="/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/train_rerank_data/minicpm_top5/marco/train_list")
    parser.add_argument('--batch_size', type=int,
                        default=1)
    parser.add_argument('--loop', type=int,
                        default=1)
    parser.add_argument('--cut_chunk', type=int,
                        default=8)
    parser.add_argument('--number_chunk', type=int,
                        default=0)
    parser.add_argument('--chat_templete', action='store_true',default=True)
    parser.add_argument('--llama_style', action='store_true',default=True)
    args = parser.parse_args()

    args = parser.parse_args()
    print("The parameter configuration is as follows:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    input = read_jsonl(args.input_path)
    split_input = split_list_evenly(input,args.cut_chunk)
    input_list = split_input[args.number_chunk]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    
    dataset = llmDataset(input_list,args,tokenizer)
    dataloader = DataLoader(dataset=dataset,
                                      batch_size=args.batch_size,collate_fn= dataset.Collactor,)
    
    temperature_list = [1.0]
    all_save_list = [[] for _ in range(len(input_list))]
    
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size= 1, trust_remote_code=True, dtype='bfloat16',)
    for temp in tqdm(temperature_list):
        params_dict = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "temperature": temp,
            "top_p": 0.8,
            "top_k": -1,
            "use_beam_search": False,
            "length_penalty": 1,
            "early_stopping": False,
            "stop": None,
            "stop_token_ids": None,
            "ignore_eos": False,
            "max_tokens": 100,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }

        # Create a sampling params object.
        sampling_params = SamplingParams(**params_dict)

        # Create an LLM.
        locate = 0
        prior = 0
        next = 0


        for batch in tqdm(dataloader):
            
            batch_augment_query = batch['augment_input']
            batch_augment_query = sum(batch_augment_query ,[])
            ids = batch['id']
            type_list = batch['type']
            data_type_list = batch['data_type']
            locate = next
            prior = next
            
            for loop in range(args.loop):
                locate = prior
                augment_outputs = llm.generate(batch_augment_query, sampling_params)
                augment_outputs = [augment_outputs[i:i + 100] for i in range(0, len(augment_outputs), 100)]
                
                for id, aug, type,data_type in zip(ids,augment_outputs, type_list,data_type_list):
                    if loop == 0 :
                        all_save_list[locate] = {}
                        all_save_list[locate]['id'] = id
                        all_save_list[locate]['context']=[]
                    
                    aug_text_list =[]
                    for aug_text in aug:
                        aug_text_list.append(aug_text.outputs[0].text)
                        
                    for idx, aug_text in enumerate(aug_text_list):
                        tempt = {}
                        tempt['text'] = aug_text
                        tempt['temperature'] = temp
                        tempt['type'] = type[idx]
                        tempt['data_type']=data_type
                        tempt['loop'] = str(loop+1)
                        all_save_list[locate]['context'].append(tempt)
                                           
                    locate+=1
                    if loop == 0:
                        # next 为下一个开始的索引
                        next = locate
                    
    out_put_path = os.path.join(args.output_path,'all_text_data_{}.jsonl'.format(args.number_chunk))
    with open(out_put_path, 'w', encoding='utf-8') as f:
        for item in all_save_list:
            f.write(json.dumps(item) + '\n')


    print("---------finish--------------")

if __name__ == "__main__":
    main()