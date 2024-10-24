cd ../src/knowledgeRefinement
gpu_vis=0,1,2,3,4,5,6,7
MASTER_PORT=2346
deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT train.py \
    --model_name_or_path # The path of LLM llama or minicpm \
    --train_data_path # The path of DPO train dataset \
    --eval_data_path # The path of DPO dev dataset \
    --max_length 1270 \
    --max_prompt_length 1256 \
    --max_passage_length 1024 \
    --output_dir # The path to save checkpoint \
    --save_steps 50 \
    --eval_steps 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_dir # The path to save log \
    --bf16 True \
    --use_lora True \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 2 \
    --deepspeed config/ds_config_zero2.json
