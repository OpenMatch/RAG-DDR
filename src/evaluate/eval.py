import argparse
import json
import os

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
from evaluate.eval_utils import load_file, process_input_data, postprocess_output, test_kilt
from vllm import LLM, SamplingParams

def call_vllm_model(args, prompts, user_chat_template, model, tokenizer, max_new_tokens,sampling_params):
    if user_chat_template:
        chat_prompts = []
        for prompt in prompts:
            if args.llama_style:
                messages = [
                    {"role": "user", "content": prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            else:
                prompt = "<用户>{}<AI>".format(prompt)
            chat_prompts.append(prompt)
        prompts = chat_prompts
    outputs = model.generate(prompts, sampling_params)

    preds = []
    for pred in outputs:
        pred = pred.outputs[0].text.lstrip()
        preds.append(pred)
    postprocessed_preds = [postprocess_output(pred) for pred in preds]
    return postprocessed_preds, prompts


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default=None)
    parser.add_argument('--input_file', type=str, 
                        default=None)
    parser.add_argument('--retrieval_augment', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--device', type=str, 
                        default="cuda")
    parser.add_argument('--max_new_tokens', type=int, 
                        default=32)
    parser.add_argument('--max_length', type=int, 
                        default=4096)
    parser.add_argument('--metric', type=str, 
                        default='accuracy')
    parser.add_argument('--top_n', type=int, 
                        default=1,help="number of passages to be considered.")
    parser.add_argument('--task', type=str, 
                        default=None,help="which task will be used.")
    parser.add_argument('--user_chat_template', action='store_true')
    parser.add_argument('--llama_style', action='store_true',
                        help="whether to use llama.")
    parser.add_argument('--rerank', action='store_true',
                        help="whether to use refinement.")

    parser.add_argument('--output_path', type=str,default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--case_num', type=int, default=-1)
    args = parser.parse_args()

    print("The parameter configuration is as follows:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    if args.output_path!=None:
        output_path = os.path.join(args.output_path, args.exp_name)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path=None

    model = LLM(model=args.model_name_or_path, tensor_parallel_size= 1, trust_remote_code=True,)     
    params_dict = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "use_beam_search": False,
            "length_penalty": 1,
            "early_stopping": False,
            "stop": None,
            "stop_token_ids": None,
            "ignore_eos": False,
            "max_tokens": args.max_new_tokens,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }

    # Create a sampling params object.
    sampling_params = SamplingParams(**params_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                                padding_side="left",truncation_side="right",is_pretrained_tokenizer=True )
    tokenizer.pad_token = tokenizer.eos_token

    # for top_n in args.top_n:
    input_data = load_file(args.input_file)
    if args.case_num != -1:
        input_data = input_data[:args.case_num]
    input_data = process_input_data(input_data, args, args.top_n, tokenizer)

    final_results = []
    for idx in tqdm(range(len(input_data))):
        batch = input_data[idx:idx+1]
        processed_batch = [item['instruction'] for item in batch]
        preds, prompts = call_vllm_model(
            args, processed_batch, user_chat_template=args.user_chat_template, model=model, tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens, sampling_params=sampling_params)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = pred
            item["prompts"] = prompts
            final_results.append(item)

    if output_path is not None:
        output_path = os.path.join(output_path, str(args.task)+'output.jsonl')
        with open(output_path, "w") as f:
            for item in input_data:
                json.dump(item, f)
                f.write("\n")
    print("results are saved in:", output_path)

    for item in input_data:
        metric_result = test_kilt(args.task,args.metric,item["output"], item)
        item["metric_result"] = metric_result

    print(args.task)
    print("overall result: {0}".format(
        np.mean([item["metric_result"] for item in input_data])))
    print('finish')

if __name__ == "__main__":
    main()