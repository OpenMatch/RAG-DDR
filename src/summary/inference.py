import argparse
import json
import os


import numpy as np

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LLMSummary:

    def __init__(
            self,
            args = None,
            model_name_or_path: str = None,
            ):
        
        model = LLM(model= model_name_or_path, tensor_parallel_size= 1, trust_remote_code=True,)
                    #dtype='bfloat16',)
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
                "max_tokens": 4096,
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
        self.sampling_params =sampling_params
        self.args = args
        self.system_prompt = """You are tasked with summarizing the context in a concise manner, focusing on the key information relevant to answering the given question. 
                                > Question: {question}
                                > Context:
                                >>>
                                {context}
                                >>>
                                > Provide a concise summary that captures the main points and relevant details from the context to address the question:"""
 
    def get_inputs(self, content, query):
        if self.args.task == "t-rex":
            query = "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity. {}\nAnswer:".format(query)

        content = self.truncated_passage(content, self.args.max_psg_length)
        input_text = self.system_prompt.format(question = query, context = content)

        if self.args.llama_style:
            messages = [
                {"role": "user", "content": input_text},
            ]
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:

            inputs = "<用户>{}<AI>".format(input_text)
  
        return inputs


    def truncated_passage(self, passage, truncate_size):
        encoded_passage = self.tokenizer.encode(passage, add_special_tokens=False)
        truncated_encoded_passage = encoded_passage[:truncate_size]
        decoded_passage = self.tokenizer.decode(truncated_encoded_passage)
        return decoded_passage
    
    def inference(self, content, query):
        inputs = self.get_inputs(content, query)
        outputs = self.model.generate(inputs, self.sampling_params)
        for pred in outputs:
            pred = pred.outputs[0].text.lstrip()
       
        return pred



def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
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





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file_path', type=str,
                        default="/data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl")
    parser.add_argument('--model_name_or_path', type=str,
                        default="/data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct")
    parser.add_argument('--output_path', type=str,
                        default="/data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval")
    parser.add_argument('--task', type=str,default='nq')
  
    parser.add_argument('--max_psg_length', type=int, default=8000)
    parser.add_argument('--top_n', type=int,
                        default=5)
    parser.add_argument('--cut_chunk', type=int,default=1)
    parser.add_argument('--number_chunk', type=int,default=0)
  
    parser.add_argument('--llama_style', action='store_true',default=True)
    
    args = parser.parse_args()
    print("模型的所有参数:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.output_path!=None:
        output_path = os.path.join(args.output_path, args.task)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path=None

    summary_agent = LLMSummary(args = args, model_name_or_path=args.model_name_or_path)

    raw_data = read_jsonl(args.dataset_file_path)
    raw_data_list = split_list_evenly(raw_data,args.cut_chunk)
    data = raw_data_list[args.number_chunk]

    for example in tqdm(data, desc="Processing examples"):
        if 'input' in example:
            question = example['input']
        elif 'query' in example:
            question = example['query']
        elif 'question' in example:
            question = example['question']

        if len(example['rerank_passage'])>= args.top_n:
            passages = example['rerank_passage'][:args.top_n]
        else:
            passages = example['rerank_passage'] + example['passage'][:args.top_n-len(example['rerank_passage'])] 
        passages_text = []
        for psg in passages:
            segment = psg['segment']
            passages_text.append(segment)
        passages_text = '\n'.join(passages_text)
      
            
        passage_summary = summary_agent.inference(passages_text, question)
        example['passage_summary'] = passage_summary
  
        
    output_path = os.path.join(output_path,'all_text_data.{}.jsonl'.format(args.number_chunk))
    with open(output_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")
    print('finish')
    

    
    

if __name__ == "__main__":
    main()