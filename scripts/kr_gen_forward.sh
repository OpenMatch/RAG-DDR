cd ../src/knowledgeRefinement
gpu=(0 1 2 3 4 5 6 7)
items=(0 1 2 3 4 5 6 7)
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    python gen_llm_response.py  \
        --input_path # The path of train/dev data \
        --model_name_or_path # The path of gen moudle Llama/Minicpm  \
        --output_path # The path to save dpo sample  \
        --top_n 100 \
        --cut_chunk 8 \
        --llama_style # If llm is llama, use it. \
        --number_chunk ${items[$i]}
done