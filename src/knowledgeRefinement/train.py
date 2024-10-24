import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments,PreTrainedModel,AutoConfig)
from functools import partial
import logging
from trl import DPOTrainer
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import load_dataset, Dataset
import torch
import transformers
import torch.nn as nn
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from peft import PeftConfig, PeftModel
from peft import LoraConfig, TaskType, get_peft_model
import random
logger = logging.getLogger(__name__)
random.seed(42)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    cache_dir: str = field(default="tmp")


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
    
    max_length: int = field(default=1270,metadata={"help":"Maximum all sequence length."},)
    max_prompt_length: int = field(default=1256,metadata={"help":"Maximum prompt prompt length."},)
    max_passage_length: int = field(default=1024,metadata={"help":"Maximum prompt passage length."},)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default=None)
    use_lora: bool = field(default=True)
    output_dir : str = field(default=None)
    save_steps : int = field(default=1000)
    eval_steps : int = field(default=200)
    per_device_train_batch_size: int = field(default=1)
    evaluation_strategy: str = field(default='steps')
    logging_steps : int = field(default=10)
    logging_dir : str = field(default=None)
    bf16 : bool = field(default=True)
    num_train_epochs: int = field(default=10)

def load_model_and_tokenizer(
    model_path: str,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = False,
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
        lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    inference_mode=False,
                )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def preprocessing(example,args,tokenizer):
        
    one_item = {}
    passage = random.choice(example['error_passage'])['segment']
    query = example['question']
    system_prompt = """Given the following question and context,
return YES if the context is relevant to the question and NO if it isn't.

> Question: {question}
> Context:
>>>
{context}
>>>
> Relevant (YES / NO):"""

    passage_inputs = tokenizer(passage,return_tensors=None,
                                        add_special_tokens=False,
                                        max_length=args.max_passage_length,
                                        truncation=True)['input_ids']
    new_passage = tokenizer.decode(passage_inputs,skip_special_tokens=True)
    new_prompt = system_prompt.format(question = query, context = new_passage)
    messages = [
            {"role": "user", "content": new_prompt},
        ]
    item_input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    one_item["prompt"] = item_input_ids
    one_item["chosen"] = example["chosen"]
    one_item["rejected"] = example["rejected"]
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
        use_lora=training_args.use_lora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
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