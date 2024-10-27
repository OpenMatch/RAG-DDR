from vllm import LLM, SamplingParams
import argparse
import json
from tqdm import  tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
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

      
class LLMDataset(Dataset):
    def __init__(self,data_list,args,tokenizer):
        self.data_list = data_list
        self.args = args
        self.tokenizer = tokenizer
        
    def process_query_with_psg(self, item):

        if self.args.llama_style:
            if item['data_type'] in ['commonsense_qa', 'math_qa',"aqua_rat","ecqa"]:
                raw_input = llama_multi_choice.format(item['question'])

            if item['data_type'] in ['coqa', 'web_questions', 'wiki_qa', 'yahoo_answers_qa',"marcoqa"]:
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
            
            if item['data_type'] in ['coqa', 'web_questions', 'wiki_qa', 'yahoo_answers_qa',"marcoqa"]:
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

        item['augment_type'] = []
        item['augment_type'].append('raw')
        item = self.get_more_psg(item,raw_input) 
        return item

    
    def get_more_psg(self,item,raw_input):
        psg_list = []

        if len(item['rerank_passage'])>=self.args.top_n:
            #item['rerank_passage'] represents the passage after we use knowledge refine
            psgs = item['rerank_passage'][:self.args.top_n]
        else:
            #Not enough n-passage to do filler.
            psgs = item['rerank_passage']+item['passage'][:self.args.top_n-len(item['rerank_passage'])]

        token_query = self.tokenizer([raw_input])
        query_length = len(token_query.input_ids[0])

        for p in psgs:
             psg_list.append(p['segment'])

        # cut too long passage
        aug_psg = '\n'.join(psg_list[:self.args.top_n])
        token_aug_psg = self.tokenizer([aug_psg ])
        token_aug_psg = token_aug_psg.input_ids[0][:4096-SPECIAL_TOKEN_LENGTH-query_length]
        new_psg = self.tokenizer.decode(token_aug_psg,skip_special_tokens=True)

        if self.args.llama_style:
            aug_input = [{"role": "user", "content": augment_templeta.format(new_psg, raw_input)}]
            aug_input = self.tokenizer.apply_chat_template(aug_input, add_generation_prompt=True, tokenize=False)
            item['augment_input']=aug_input
        
        else:
            item['augment_input']=minicpm_prompt_template.format(augment_templeta.format(new_psg, raw_input))
        item['augment_type'].append('aug_1-5')
            
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
        
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str,
                        default=None,
                        help="The path of the training/evaluation data file to be processed.",
                        )
    parser.add_argument('--model_name_or_path', type=str,
                        default=None,
                        help="The path of the LLM.")
    parser.add_argument('--output_path', type=str,
                        default=None,
                        help="The path of the DPO data generated by LLM.")
    parser.add_argument('--batch_size', type=int,
                        default=8)
    parser.add_argument('--top_n', type=int,
                        default=5)
    parser.add_argument('--llama_style', action='store_true',
                        help="Whether to use the Llama model.")
    parser.add_argument('--loop', type=int,
                        default=5,
                        help="The number of multiple sampling rounds.",
                        )
    parser.add_argument('--cut_chunk', type=int,
                        default=8,
                        help="The number of data segments.",
                        )
    parser.add_argument('--number_chunk', type=int,
                        default=0,
                        help="The current index of data segments.",
                        )
    args = parser.parse_args()
    
    
    print("The parameter configuration is as follows:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    input = read_jsonl(args.input_data_path)
    split_input = split_list_evenly(input,args.cut_chunk)
    input_list = split_input[args.number_chunk]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = LLMDataset(input_list,args,tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn= dataset.Collactor,)
    
    temperature_list = [0.5,0.6,0.7,0.8,0.9]
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

            batch_raw_query = batch['raw_input']
            batch_augment_query = batch['augment_input']
            ids = batch['id']
            type_list = batch['type']
            type_list = sum(type_list,[])
            # Both locate and prior are reset
            locate = next
            prior = next
                      
            for loop in range(args.loop):
                # Each time a large loop starts, locate is reset to prior
                locate = prior
                merged_input = []
                for a, b in zip(batch_raw_query,batch_augment_query):
                    merged_input.append(a)
                    merged_input.append(b)

                outputs = llm.generate(merged_input, sampling_params)
                ids=[item for pair in zip(ids, ids) for item in pair]
                for id, out, type in zip(ids,outputs, type_list):
                    # Only the first large loop needs to put data, so here loop=0
                    if all_save_list[locate]==[]:
                        all_save_list[locate] = {}
                        all_save_list[locate]['id'] = id
                        all_save_list[locate]['context']=[]
                    
                    tempt = {}
                    tempt['text'] = out.outputs[0].text
                    tempt['temperature'] = temp
                    tempt['type'] = type
                    tempt['loop'] = str(loop+1)
                    all_save_list[locate]['context'].append(tempt)

                    if type == 'aug_1-5':
                        locate+=1

                    if loop == 0:
                        # next is the next starting index
                        next = locate
                    
    out_put_path = os.path.join(args.output_path,'all_text_data.{}.jsonl'.format(args.number_chunk))
    with open(out_put_path, 'w', encoding='utf-8') as f:
        for item in all_save_list:
            f.write(json.dumps(item) + '\n')

    print("---------finish--------------")

if __name__ == "__main__":
    main()


