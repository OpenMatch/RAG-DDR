
torchrun --nproc_per_node 8 -m openmatch.driver.build_index \
 --output_dir  # The path to save marco2.0 embedding \
 --model_name_or_path # The path of bge model \
 --per_device_eval_batch_size 2048 \
 --corpus_path # The path of marco2.0 corpus \
 --doc_template "<segment>" \
 --q_max_len 64 \
 --p_max_len 256 \
 --max_inmem_docs 1000000