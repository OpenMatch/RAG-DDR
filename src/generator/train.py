import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments)

from functools import partial
import logging
from trl import DPOTrainer
import transformers
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
import torch
import transformers

logger = logging.getLogger(__name__)


MAX_PROMPT_LENGTH = 32

prompt_template = "<用户>{}<AI>"
augment_templeta = 'Background:\n{}\n\n{}'


multi_choice = 'The following is multiple choice question. Please choose the best answer choice which can answer the following question.\n{}\nAnswer:'
QA_templeta = 'Q: {}\nA:'

Mult_COT_templeta = "Please answer multiple choice question and choose the best answer choice first. Then give your explanation between [<COT] and [COT>]."
QA_COT_templeta = "Please answer the question. Then give your explanation between [<COT] and [COT>]."

COT_few_shot = 'question: {}\nAnswer:{}\n[<COT] {} [COT>]\n '
COT_question = 'question: {}\nAnswer:'


llama_multi_choice = 'Please answer the multiple choice questions below and output only the choice.\n{}\nAnswer:'
llama_QA_COT_templeta = "Please answer the question and only output the answer. Then give your explanation between [<COT] and [COT>]."


en_system_prompt = "Answer the question based on the given document. \
    Please include the relevant reasoning process and key information. \
    The following are the given documents:\n\n{reference}"

en_user_prompt = "Question: {question}\nBased on the provided documents, \
    combine the question to give the answer, \
    including the intermediate reasoning process and key information:"

zh_system_prompt = "基于给定文档回答问题。请包含相关的推理过程和相关信息。以下是给定文档，你需要根据问题输出回答：\n\n{reference}"
zh_user_prompt = "问题：{question}\n根据上面提供的文档，结合问题给出答案，请包含中间推理过程和关键信息："

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llama_style: bool = field(default=False)
    use_template: bool = field(default=True)


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the test data."},
    )
    
    max_length: int = field(default=600,metadata={"help":"Maximum all sequence length."},)
    max_prompt_length: int = field(default=384,metadata={"help":"Maximum prompt sequence length."},)
    
    top_n: int = field(default=5,metadata={"help":"how many psg use."},)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_lora_model : bool = field(default=True)

    use_lora: bool = field(default=True)
    output_dir : str = field(default=None)
    save_steps : int = field(default=10)
    eval_steps : int = field(default=100)
    per_device_train_batch_size: int = field(default=4)
    evaluation_strategy: str = field(default='steps')
    logging_steps : int = field(default=10)
    logging_dir : str = field(default=None)
    bf16 : bool = field(default=True)



def load_model_and_tokenizer(
    model_path: str,
    llama_style: bool,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = False,
    load_lora_model: bool = False,
):
    """load model and tokenizer"""


    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    tokenizer.pad_token = tokenizer.eos_token
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        if llama_style:   
            lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    inference_mode=False,
                )
        else:
            lora_config = LoraConfig(
                init_lora_weights="gaussian",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"],

                # target_modules=['q_a_proj',
                #     'q_b_proj',
                #     'kv_a_proj_with_mqa',
                #     'kv_b_proj',
                #     'o_proj',],
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                inference_mode=False,
            )
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer

def format_reference(language, retrieval_result):
    format_reference = ''

    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item
        if language == 'en':
            format_reference += f"Doc {idx+1}: {content}\n"
        else:
            format_reference += f"文档{idx+1}：{content}\n"

    return format_reference

def preprocessing(example,args,tokenizer):
        one_item = {}

        if model_args.llama_style:

            if example['data_type'] in ['commonsense_qa', 'math_qa',"aqua_rat","ecqa"]:
                query = llama_multi_choice.format(example['question'])

            if example['data_type'] == 'coqa' or example['data_type'] =='web_questions' or example['data_type'] =='wiki_qa' or example['data_type'] =='yahoo_answers_qa' \
                or example['data_type'] =="marcoqa":
                query = QA_templeta.format(example['question'])
            
            if example['data_type'] in ["gsm8k","strategyqa"]:
                query = llama_QA_COT_templeta + COT_question.format(example['question']) 

            if example['data_type'] in ['rag_bench']:
                query = example['question']    

        else:
               
            if example['data_type'] == 'commonsense_qa' or example['data_type'] == 'math_qa':
                query = multi_choice.format(example['question'])
                
            if example['data_type'] == 'coqa' or example['data_type'] =='web_questions' or example['data_type'] =='wiki_qa' or example['data_type'] =='yahoo_answers_qa' \
                or example['data_type'] =="marcoqa":
                query = QA_templeta.format(example['question'])
            
            if example['data_type'] in ["aqua_rat","ecqa"]:
                query = Mult_COT_templeta + COT_question.format(example['question'])
            
            if example['data_type'] in ["gsm8k","strategyqa"]:
                query = QA_COT_templeta + COT_question.format(example['question'])

            if example['data_type'] in ['rag_bench']:
                query = example['question'] 
                formatted_reference = format_reference(example['language'],example['passage'])

                input_params = {
                    "question": query,
                    "reference": formatted_reference
                }

                if example['language']=='en':
                    system_prompt = en_system_prompt.format(**input_params)
                    user_prompt = en_user_prompt.format(**input_params)
                
                else:
                    system_prompt = zh_system_prompt.format(**input_params)
                    user_prompt = zh_user_prompt.format(**input_params)
                new_aug_psg = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])
                aug_query = prompt_template.format(new_aug_psg)

                
        if example['data_type'] not in ['rag_bench']:

            if len(example['rerank_passage'])>= args.top_n:
                psgs = example['rerank_passage'][:args.top_n]
            else:
                psgs = example['rerank_passage']+example['passage'][:args.top_n-len(example['rerank_passage'])]             

            # psgs = example['rerank_passage'][:args.top_n]

            psg_list = []

            for p in psgs:
                if isinstance(p, str):
                    psg_list.append(p)

                else:
                    if 'passage_text' in p:
                        psg_list.append(p['passage_text'])
                        
                    elif "segment" in p:
                        psg_list.append(p['segment'])

            aug_psg = '\n'.join(psg_list)
            
            token_query = tokenizer([query])
            query_length = len(token_query.input_ids[0])
            
            token_aug_psg = tokenizer([aug_psg])
            token_aug_psg = token_aug_psg.input_ids[0][:args.max_prompt_length-MAX_PROMPT_LENGTH-query_length]
            new_aug_psg = tokenizer.decode(token_aug_psg,skip_special_tokens=True)
        
            if model_args.llama_style:

                if model_args.use_template:
                    input_data = augment_templeta.format(new_aug_psg, query)
                    aug_query = [{"role": "user", "content": input_data},]
                    aug_query = tokenizer.apply_chat_template(aug_query, add_generation_prompt=True, tokenize=False)
                
                else:
                    input_data = augment_templeta.format(new_aug_psg, query)
                    aug_query = input_data

            else:
                aug_query = prompt_template.format(augment_templeta.format(new_aug_psg, query))

        one_item["prompt"] = aug_query
        one_item["chosen"] = example["chosen"]['text']
        one_item["rejected"] = example["rejected"]['text']

        return one_item

if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)


    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        llama_style = model_args.llama_style,
        use_lora=training_args.use_lora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        load_lora_model =training_args.load_lora_model
    )
    
    partial_preprocess = partial(preprocessing,args=data_args,tokenizer=tokenizer)

    train_dataset = load_dataset("json", data_files=data_args.train_data_path,split="train",)
    train_dataset = train_dataset.map(partial_preprocess)

    eval_dataset = load_dataset("json", data_files=data_args.eval_data_path,split="train",)
    eval_dataset = eval_dataset.map(partial_preprocess)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset =eval_dataset,
        max_length = data_args.max_length,
        max_prompt_length = data_args.max_prompt_length,
        tokenizer=tokenizer,

    )
    dpo_trainer.train()
    dpo_trainer.save_model()




