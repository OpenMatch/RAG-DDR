import argparse
import json
import os
import pdb
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


class LLMReranker:
    def __init__(
            self,
            args = None,
            model_name_or_path: str = None,
            topn: int = 10,
            needn:int =5):
        
        model = LLM(model= model_name_or_path, tensor_parallel_size= 1, trust_remote_code=True,)
        params_dict = {
                "n": 1,
                "best_of": 1,
                "presence_penalty": 1.0,
                "frequency_penalty": 0.0,
                "temperature": 0.8,
                "top_p": 1.0,
                "top_k": 1,
                "use_beam_search": False,
                "length_penalty": 1,
                "early_stopping": False,
                "stop": None,
                "stop_token_ids": None,
                "ignore_eos": False,
                "max_tokens": 32,
                "logprobs": None,
                "prompt_logprobs": None,
                "skip_special_tokens": True,
            }

        # Create a sampling params object.
        sampling_params = SamplingParams(**params_dict)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                                    padding_side="left",truncation_side="right",is_pretrained_tokenizer=True )
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer =tokenizer

        self.model = model
        self.topn = topn
        self.sampling_params =sampling_params
        self.needn = needn
        self.args = args
        self.system_prompt = """Given the following question and context,
return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):"""
   

    def get_inputs(self, pairs, prompt=None, max_length=1024):
        """Build input tokens with query and chunks."""
        inputs = []
        for query, passage in pairs:
            
            if self.args.task == "t-rex":
                query = "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity. {}\nAnswer:".format(
                    query)

            passage_inputs = self.tokenizer(passage,
                                            return_tensors=None,
                                            add_special_tokens=False,
                                            max_length=max_length,
                                            truncation=True)['input_ids']
            new_passage = self.tokenizer.decode(passage_inputs,skip_special_tokens=True)
            new_prompt = self.system_prompt.format(question = query, context = new_passage)
            messages = [
                    {"role": "user", "content": new_prompt},
            ]
            item_input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs.append(item_input_ids)

        return inputs

    def split_list(self,input_list, n):
        return [input_list[i:i + n] for i in range(0, len(input_list), n)]
    
    def judege_output(self,pred):
        return 'YES' in pred or 'yes' in pred or 'Yes' in pred 

    def inference(self, chunks, query):
        pairs = []
        for chunk in chunks:
            pairs.append([query, chunk])
        split_chunks = self.split_list(pairs,self.needn)
        count_of_ones = 0
        judge_preds = []
        preds = []

        for sub_chunk in split_chunks:

            inputs = self.get_inputs(sub_chunk)
            outputs = self.model.generate(inputs, self.sampling_params)
            for pred in outputs:
                pred = pred.outputs[0].text.lstrip()
                judge_preds.append(self.judege_output(pred))
                if self.judege_output(pred) == True:
                    count_of_ones +=1
                preds.append(pred)

            if count_of_ones >= self.needn:
                break
        return judge_preds

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def split_list(lst, n):
    array = np.array(lst)
    return np.array_split(array, n)


def get_batch_input(data, batch_size):
    split_data =[]
    split_array = split_list(data, np.ceil(len(data) / batch_size))
    for arr in split_array:
        split_data.append(arr.tolist())

    return split_data


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file_path', type=str,
                        default=None)
    parser.add_argument('--model_name_or_path', type=str,
                        default=None)
    parser.add_argument('--output_path', type=str,
                        default=None)
    parser.add_argument('--file_name', type=str,
                        default=None)
    parser.add_argument('--top_n', type=int,
                        default=100)
    parser.add_argument('--need_n', type=int,
                        default=5)
    parser.add_argument('--cut_num', type=int,default=1)
    parser.add_argument('--number', type=int,default=0)
    parser.add_argument('--batch_size', type=int,default=3)
    parser.add_argument('--task', type=str,default=None)
    
    args = parser.parse_args()
    print("模型的所有参数:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    reranker = LLMReranker(args = args, model_name_or_path=args.model_name_or_path, topn=args.top_n,needn=args.need_n)

    raw_data = read_jsonl(args.dataset_file_path)
    raw_data_list = split_list(raw_data, args.cut_num)
    data = raw_data_list[args.number].tolist()
    data = get_batch_input(data, args.batch_size)

    for item in tqdm(data, desc="Processing examples"):

        for example in item:
            if 'input' in example:
                question = example['input']
            elif 'query' in example:
                question = example['query']
            elif 'question' in example:
                question = example['question']

            passages = []
            for psg in example['passage']:
                segment = psg['segment']
                passages.append(segment)
            passages = passages[:args.top_n] 
            judge_preds = reranker.inference(chunks=passages, query=question)
            example['judge_preds'] = judge_preds
  
        for example in item:
            example['rerank_passage'] = []
            if any(example['judge_preds']):
                for judge,psg in zip(example['judge_preds'],example['passage']):
                    if judge:
                        example['rerank_passage'].append(psg)
            else:
                example['rerank_passage'] = example['passage']
      
    data = sum(data, [])
    output_path = os.path.join(args.output_path, args.file_name)
    with open(output_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")
    
if __name__ == "__main__":
    main()