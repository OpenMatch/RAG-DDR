cd ../src/evaluate
gpu=(2)
items=(5)
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    python eval.py  \
        --model_name_or_path # the path of gen model  \
        --input_file # the path of test data \
        --retrieval_augment # if you test in w/ rag, use --retrieval_augment \
        --use_lora \
        --max_new_token 32 # ms marco dataset is 100, other dataset is 32 \
        --metric accuracy # the metrics identifiers \
        --task t-rex # the task identifiers \
        --top_n ${items[$i]}  \
        --rerank \
        --llama_style # if use llama as gen moudel, llama_style is used, else not \
        --user_chat_template
done