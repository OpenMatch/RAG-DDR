python /openmatch/driver/retrieve.py \
 --output_dir # The path to save marco2.0 embedding and query embedding \
 --model_name_or_path # The path of bge model \
 --per_device_eval_batch_size 512 \
 --query_path # The path of query data \
 --query_template "<input>" # The key name used to represent the query \
 --trec_save_path # The path to save trec \
 --q_max_len 128 \
 --retrieve_depth 100 # Top n passage to retrieve \
 --use_gpu True