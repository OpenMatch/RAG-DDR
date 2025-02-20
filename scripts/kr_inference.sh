cd ../src/knowledgeRefinement
gpu=(0 1 2 3)
items=(0 1 2 3)
length=${#items[@]}
for ((i=0; i<$length; i++)); do
    export CUDA_VISIBLE_DEVICES=${gpu[$i]}
    python kr_inference.py \
        --dataset_file_path # the dataset which need be refined  \
        --model_name_or_path # the path of knowledge refinement moudle \
        --output_path # the path to save checkpoint  \
        --file_name "${items[$i]}".jsonl  # save file name\
        --cut_num 4 \
        --batch_size 1 \
        --number ${items[$i]} \
        --need_n 5 # Number of documents to be retained \
        --top_n 100 
done