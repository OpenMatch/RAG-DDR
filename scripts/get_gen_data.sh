cd ../src/generator
python gen_dpo_data.py  \
    --raw_input_path # The path of constructed train/dev data \
    --dpo_input_path # The path of DPO sample generated by gen_llm_response.py \
    --out_path # The path save the DPO train data \
    --llama_style # If gen module is llama, use it.
