
export CUDA_VISIBLE_DEVICES=0
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task nq  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --llama_style  > nq0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task nq  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --llama_style  > nq1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task nq  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --llama_style  > nq2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task nq  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --llama_style  > nq3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task t-rex  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --llama_style  > trex0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task t-rex  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --llama_style  > trex1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task t-rex  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --llama_style  > trex2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task t-rex  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --llama_style  > trex3.out  2>&1 &


export CUDA_VISIBLE_DEVICES=0
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task tqa  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --llama_style  > tqa0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task tqa  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --llama_style  > tqa1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task tqa  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --llama_style  > tqa2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task tqa  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --llama_style  > tqa3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task marco  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --llama_style  > marco0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task marco  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --llama_style  > marco1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task marco  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --llama_style  > marco2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task marco  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --llama_style  > marco3.out  2>&1 &


export CUDA_VISIBLE_DEVICES=0
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task hotpotqa  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --llama_style  > hotpot0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task hotpotqa  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --llama_style  > hotpot1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task hotpotqa  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --llama_style  > hotpot2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task hotpotqa  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --llama_style  > hotpot3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task wow  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --llama_style  > wow0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task wow  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --llama_style  > wow1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task wow  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --llama_style  > wow2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/llama3/Meta-Llama-3-8B-Instruct \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval \
    --task wow  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --llama_style  > wow3.out  2>&1 &



export CUDA_VISIBLE_DEVICES=0
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task nq  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --max_psg_length 3900  > nq0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task nq  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --max_psg_length 3900  > nq1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task nq  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --max_psg_length 3900  > nq2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/nq_list/tempt/nq_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task nq  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --max_psg_length 3900  > nq3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task t-rex  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --max_psg_length 3900  > trex0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task t-rex  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --max_psg_length 3900  > trex1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task t-rex  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --max_psg_length 3900  > trex2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/trex_intro_list/4-split/trex_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task t-rex  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --max_psg_length 3900  > trex3.out  2>&1 &


export CUDA_VISIBLE_DEVICES=0
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task tqa  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --max_psg_length 3900  > tqa0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task tqa  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --max_psg_length 3900  > tqa1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task tqa  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --max_psg_length 3900  > tqa2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/tqa_list/4-split/tqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task tqa  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --max_psg_length 3900  > tqa3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=4
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task marco  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --max_psg_length 3900  > marco0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task marco  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --max_psg_length 3900  > marco1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task marco  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --max_psg_length 3900  > marco2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/marco_qa_list/4-split/marco_qa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task marco  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --max_psg_length 3900  > marco3.out  2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task hotpotqa  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --max_psg_length 3900  > hotpot0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task hotpotqa  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --max_psg_length 3900  > hotpot1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task hotpotqa  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --max_psg_length 3900  > hotpot2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=3
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/hotpotqa_list/4_split/hotpotqa_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task hotpotqa  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --max_psg_length 3900  > hotpot3.out  2>&1 &


export CUDA_VISIBLE_DEVICES=4
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task wow  \
    --cut_chunk 4  \
    --number_chunk 0  \
    --max_psg_length 3900  > wow0.out  2>&1 &

export CUDA_VISIBLE_DEVICES=5
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task wow  \
    --cut_chunk 4  \
    --number_chunk 1  \
    --max_psg_length 3900  > wow1.out  2>&1 &

export CUDA_VISIBLE_DEVICES=6
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task wow  \
    --cut_chunk 4  \
    --number_chunk 2  \
    --max_psg_length 3900  > wow2.out  2>&1 &

export CUDA_VISIBLE_DEVICES=7
nohup python inference.py  \
    --dataset_file_path /data/groups/QY_LLM_Other/lixinze/icrl_2024/rerank/LLM_rerank/marco/wow_list/4-split/wow_psg.jsonl  \
    --model_name_or_path /data/groups/QY_LLM_Other/meisen/pretrained_model/MiniCPM-2B-sft-bf16 \
    --output_path /data/groups/QY_LLM_Other/meisen/iclr2024/summary/zeroshot_eval_minicpm \
    --task wow  \
    --cut_chunk 4  \
    --number_chunk 3  \
    --max_psg_length 3900  > wow3.out  2>&1 &