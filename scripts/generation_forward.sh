cd ../src/generator
gpu=(0 1 2 3 4 5 6 7)
items=(0 1 2 3 4 5 6 7)
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    python gen_llm_response.py  \
        --input_data_path # The path of constructed orginal train/dev data  \
        --model_name_or_path # The path of LLM Llama/Minicpm  \
        --output_path # The path to save dpo sample  \
        --loop 5 \
        --top_n 5 \
        --cut_chunk 8 \
        --batch_size 8 \
        --llama_style # If llm is llama, use it. \ 
        --number_chunk ${items[$i]}
done