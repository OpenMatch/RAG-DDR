cd ../src/evaluate
gpu=(2)
items=(5)
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    nohup python eval.py  \
        --model_name_or_path /home/lixz23/ragsft/RAG-DDR_github_test/test_data/train_llama/check/merge_check  \
        --input_file /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl \
        --retrieval_augment \
        --use_lora \
        --max_new_tokens 32  \
        --metric accuracy  \
        --task t-rex  \
        --top_n ${items[$i]}  \
        --rerank \
        --llama_style \
        --user_chat_template  > /home/lixz23/ragsft/RAG-DDR_github_test/test_data/test_log/llama_t-rex_dpo_"${items[$i]}"psg.out  2>&1 &
done