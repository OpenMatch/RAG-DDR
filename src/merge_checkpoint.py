import argparse
import json
import os

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, 
                    default=None)
parser.add_argument("--save_path", type=str,
                    default=None)

args = parser.parse_args()
config = PeftConfig.from_pretrained(args.model_name_or_path)
base_tokenizer =  AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
model = PeftModel.from_pretrained(model, args.model_name_or_path)

model = model.merge_and_unload()
model.save_pretrained(args.save_path)
base_tokenizer.save_pretrained(args.save_path)