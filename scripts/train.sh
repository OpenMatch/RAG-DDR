gpu_vis=0,1,2,3,4,5,6,7
MASTER_PORT=2346
nohup deepspeed  --include localhost:$gpu_vis --master_port $MASTER_PORT /home/lixz23/ragsft/RAG-DDR/src/generator/train.py \
    --model_name_or_path /home/lixz23/pretrain-model/Llama3-8b-instruct \
    --train_data_path /home/lixz23/ragsft/RAG-DDR/data/test_data/train_llama/train.jsonl \
    --eval_data_path /home/lixz23/ragsft/RAG-DDR/data/test_data/train_llama/dev.jsonl \
    --max_length 2000 \
    --max_prompt_length 1900 \
    --output_dir /home/lixz23/ragsft/RAG-DDR/data/test_data/train_llama/check/check \
    --save_steps 100 \
    --eval_steps 100 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-5 \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --logging_steps 10 \
    --logging_dir /home/lixz23/ragsft/RAG-DDR/data/test_data/train_llama/check/log \
    --bf16 True \
    --use_lora True \
    --num_train_epochs 1 \
    --top_n 5 \
    --llama_style True \
    --deepspeed /home/lixz23/ragsft/RAG-DDR/scripts/ds_config_zero2.json > /home/lixz23/ragsft/RAG-DDR/scripts/run.log 2>&1 &

