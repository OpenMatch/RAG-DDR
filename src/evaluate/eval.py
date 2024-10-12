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
from evaluate.eval_utils import load_file, process_input_data, postprocess_output, test_kilt_em, match
from vllm import LLM, SamplingParams

CHOICE_TASK =['arc','hellaswag','socialiqa','piqa',]
KILT_TASK = ['fever','aida','t-rex','eli5','hotpotqa','wow','nq','marco','tqa','musique','wiki']

def call_model(args, prompts, user_chat_template, model, tokenizer, max_new_tokens=100):
    if user_chat_template:
        chat_prompts = []
        for prompt in prompts:
            if args.llama_style:
                messages = [
                    {"role": "user", "content": prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            elif args.cpm3:
                messages = [
                    {"role": "user", "content": prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt = "<用户>{}<AI>".format(prompt)
            chat_prompts.append(prompt)
        prompts = chat_prompts

    inputs = tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = model.generate(input_ids=input_ids.cuda(),
                             attention_mask=attention_mask.cuda(),
                             max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id,
                             top_k=1
                             )
    text_list = []
    for ids, tokens in enumerate(input_ids):
        text = tokenizer.decode(outputs[ids], skip_special_tokens=True)
        actual_prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        text_list.append(text[len(actual_prompt):])

    preds = []
    for pred in text_list:
        pred = pred.lstrip()
        # notice:这里改了
        # preds.append(pred.split("\n")[0])
        preds.append(pred)

    postprocessed_preds = [postprocess_output(pred) for pred in preds]
    return postprocessed_preds, preds

def call_vllm_model(args, prompts, user_chat_template, model, tokenizer, max_new_tokens,sampling_params):


    if user_chat_template:
        chat_prompts = []
        for prompt in prompts:
            if args.llama_style:
                messages = [
                    {"role": "user", "content": prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            elif args.cpm3:
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
        # notice:这里改了
        # preds.append(pred.split("\n")[0])
        preds.append(pred)

    postprocessed_preds = [postprocess_output(pred) for pred in preds]

    return postprocessed_preds, prompts



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                        default=None)
    parser.add_argument('--input_file', type=str, default=None)
    parser.add_argument('--retrieval_augment', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=4096)

    parser.add_argument('--metric', type=str, default='accuracy')
    parser.add_argument('--top_n', type=int, default=1,help="number of paragraphs to be considered.")
    parser.add_argument('--task', type=str, default='wiki')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--user_chat_template', action='store_true')

    parser.add_argument('--replug', action='store_true')
    parser.add_argument('--llama_style', action='store_true')
    parser.add_argument('--vllm', action='store_true')
    parser.add_argument('--cpm3', action='store_true')
    parser.add_argument('--rerank', action='store_true')

    parser.add_argument('--output_path', type=str,
                        default=None)
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


    if args.vllm:
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

    else:
        if args.use_lora:
            config = PeftConfig.from_pretrained(args.model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    trust_remote_code=True,
                )
            model = PeftModel.from_pretrained(model, args.model_name_or_path).cuda()
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left",
                                                    truncation_side="right", )
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left",
                                                    truncation_side="right", )
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map='cuda', trust_remote_code=True)


        tokenizer.pad_token = tokenizer.eos_token
        model.eval()


    ##### for top_n in args.top_n:

    input_data = load_file(args.input_file)
    if args.case_num != -1:
        input_data = input_data[:args.case_num]

    input_data = process_input_data(input_data, args, args.top_n, tokenizer)

    print("-----------case---------------")
    print(input_data[0])
    print("-----------case---------------")

    final_results = []
    
    if args.vllm:
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

    else:
        for idx in tqdm(range(len(input_data) // args.batch_size)):
            batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
            processed_batch = [item['instruction'] for item in batch]
            preds, _ = call_model(
                args, processed_batch, user_chat_template=args.user_chat_template, model=model, tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens)
            for j, item in enumerate(batch):
                pred = preds[j]
                item["output"] = pred
                final_results.append(item)

        if len(input_data) % args.batch_size > 0:
            batch = input_data[(idx + 1) * args.batch_size:]
            processed_batch = [item['instruction'] for item in batch]
            preds, _ = call_model(
                args, processed_batch, user_chat_template=args.user_chat_template, model=model, tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens)
            for j, item in enumerate(batch):
                pred = preds[j]
                item["output"] = pred
                final_results.append(item)

    if output_path is not None:
        output_path = os.path.join(output_path, str(args.task)+'output.jsonl')
        with open(output_path, "w") as f:
            for item in input_data:
                json.dump(item, f)
                f.write("\n")
    print("results are saved in:", output_path)

    for item in input_data:
        if args.task in KILT_TASK:
            metric_result = test_kilt_em(args.task,args.metric,item["output"], item)
        else:
            if args.metric == "accuracy":
                if args.task in CHOICE_TASK:
                    metric_result = item["output"].strip().startswith(item["golds"][0])
                else:
                    metric_result = 0.0
                    for pa in item["golds"]:
                        metric_result = 1.0 if pa in item["output"] or pa.lower() in item["output"] or pa.capitalize() in item["output"] else 0.0
            elif args.metric == "match":
                metric_result = match(item["output"], item["golds"])

            else:
                raise NotImplementedError
        item["metric_result"] = metric_result

    print(args.task)
    print("overall result: {0}".format(
        np.mean([item["metric_result"] for item in input_data])))
    print('finish')

if __name__ == "__main__":
    main()