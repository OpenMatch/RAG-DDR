cd ../src/knowledgeRefinement
gpu=(0 1 2 3)
items=(0 1 2 3)
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    python kr_inference.py \
        --dataset_file_path /data/home/lixz23/ragsft/data/marco_v2.1/bge_large_retriever_128_256_top100/nq_dev_psg.jsonl  \
        --model_name_or_path /home/lixz23/ragsft/DPO/icrl2024_checkpoint/LLM_rerank/doubel_train/llama_double_merge_2200_reward/merg-1000 \
        --output_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/double_train_rerank/double_llama_2300_reward_random_100/merge_1000_test_data/wow_list  \
        --file_name nq_psg_"${items[$i]}".jsonl  \
        --cut_num 4 \
        --batch_size 1 \
        --number ${items[$i]} \
        --need_n 5 \
        --top_n 100
done