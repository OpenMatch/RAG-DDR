cd ../src/generator
gpu_vis=0,1,2,3,4,5,6,7
MASTER_PORT=2346
deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT train.py \
    --model_name_or_path # The path of LLM llama or minicpm \
    --train_data_path # The path of DPO train dataset \
    --eval_data_path # The path of DPO dev dataset \
    --max_length 2000 \
    --max_prompt_length 1900 \
    --output_dir # The path to save checkpoint \
    --save_steps 100 \
    --eval_steps 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-5 \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_dir # The path to save log \
    --bf16 True \
    --use_lora True \
    --num_train_epochs 1 \
    --top_n 5 # n passage use \
    --llama_style True # if use llama as gen moudel, llama_style is True, else llama_style is False \
    --deepspeed config/ds_config_zero2.json