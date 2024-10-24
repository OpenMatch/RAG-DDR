# RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards

Source code for our paper :  
[RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards](https://arxiv.org/abs/2410.13509)

If you find this work useful, please cite our paper and give us a shining star 🌟
```
@article{li2024rag,
  title={RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards},
  author={Li, Xinze and Mei, Sen and Liu, Zhenghao and Yan, Yukun and Wang, Shuo and Yu, Shi and Zeng, Zheni and Chen, Hao and Yu, Ge and Liu, Zhiyuan and others},
  journal={arXiv preprint arXiv:2410.13509},
  year={2024}
}
```

## Requirement
**Install the following packages using Pip or Conda under this environment**

```
Python==3.10.14
torch==2.2.1
transformers==4.40.0
faiss==1.8.0
tqdm
trl==0.8.6
vllm==0.4.1
accelerate==0.30.1
deepspeed==0.14.2
peft==0.10.0
```

## Traing RAG-DDR
You can download the lora checkpoint of RAG-DDR directly from [here](https://huggingface.co/OpenMatch/RAG-DDR/tree/main) and merge them, or follow the flow below to train RAG-DDR. <br>
❗️Note: In the checkpoint file, ``Gen_model_Llama3_8b`` and ``Kr_model_for_Llama3_8b`` are the Genation and Knowledge Refinement moudle aligned by DDR training, where Llama3-8b is used as the Genation moudle. ``Gen_model_Minicpm_2.4b`` and ``Kr_model_for_Minicpm_2.4b`` are the Genation moudle and Knowledge Refinement aligned by DDR training, where Minicpm-2.4b is used as the Genation moudle.

### Prepare the training and test data
(1) Use `git clone` to download this project:
```bash
git clone https://github.com/OpenMatch/RAG-DDR.git
cd RAG-DDR
```
(2) Construct ``original train/dev dataset``:<br>
In order to construct the train/dev dataset for RAG-DDR, you can follow Table 5 in the paper to collect the corresponding dataset and process the data into the following jsonl format. Besides, you can also collect more datasets on your own and process them for DDR training.

```
{
  'id': str # The id of the data, you need to reassign a new id to the data you collected, with values from '0' to 'n-1' and 'n' being the total number of data.
  'question': str # The question of data.
  'answer': str # The ground truth of data.
  'data_type': str # Which task this data belongs to.
  'cot': str/None # Does this data have a labeled COT, if so it is provided, if not it is None.
}
```

After that, you can use [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) to retrieve relevant documents from MS MARCO 2.1 for the above constructed data. The MS MARCO 2.1 corpus can be downloaded from [here](https://trec-rag.github.io/annoucements/2024-corpus-finalization/)(❗️Note: The file name is ``msmarco_v2.1_doc_segmented.tar``). Specifically，you can go to the ``data`` folder to retrieve the relevant documents for the corresponding data. You need to use bge-large to encode all the documents as embedding and saved in ``--output_dir``:

```
cd data
bash getembedding.sh
```
Then, you need to get the appropriate trec file for the original train/dev dataset which are saved in ``--query_path`` and save it in ``--trec_save_path``:

```
bash retrieve.sh
```
Finally, you can retrieve the relevant documents for each piece of data based on the ``--trec_save_path`` and save the ``original train/dev dataset`` with the retrieved documents in ``--output_path``:

```
bash construct_psg_data.sh
```
(3) Construct ``original testing dataset``:<br>
For constructing the ``original testing dataset``, you can download KILT's test dataset from [here](https://github.com/facebookresearch/KILT) and select the corresponding test dataset from KILT according to Table 5 in the paper. After that, you can retrieve the relevant documents for each test dataset to get ``original testing dataset`` based on the above data processing steps.

(4) Download the constructed data directly:<br>
We also provide the [``original train/dev dataset``](https://drive.google.com/drive/folders/1c67ei4Lx2mC0U-dMcHtLbS5oEXoDF8np?usp=drive_link) and [``original testing dataset``](https://drive.google.com/drive/folders/1bvIdpTWi12lR_WoMfO6fAwukOfjJeIE1?usp=drive_link), which we have processed.

### Training the Generation Model
After constructing the training and test data, you can start training the RAG-DDR model. You need to train the Generation Model in RAG-DDR by constructing DPO train/dev dataset with the previously constructed ``original train/dev dataset``.

(1) First step: You need to download [Minicpm-2.4B-sft](https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16) and [lama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model as the vanilla Generation Model and lama3-8B-Instruct as Knowledge Refinement Model.

(2) Second step: You need to pass the ``original train/dev dataset`` into the vanilla Knowledge Refinement Module to filter out the retrieved external documents that are not relevant to the query and save the datastes with refinement documents in ``--output_path``. (❗️Note: You only need to use the Knowledge Refinement to retain top-5 documents for subsequent generation of the Generation module, so just set ``--need_n`` to 5 to reduce inference time.
```
cd scripts
bash kr_inference.sh
```

(3) Third step: Start generating DPO sampling data based on the vanilla Generation Model and refined dataset from second step.
```
bash gen_forward.sh
```
(4) Fourth step: The data files generated in third step are merged using ``merge_response_file.py`` to obtain DPO sampling data. Then generate DPO train/dev dataset to train Generation Model based on these DPO samples.

```
bash get_gen_data.sh
```
(5) Fifth step: Use the DPO train/dev dataset from fourth step to train Generation Modul, you can choose Minicpm-2.4B-sft or lama3-8B-Instruct model as Generation Modul.
```
bash train_gen.sh
```
(6) Sixth step: Combine the weights of the Generation Modul trained using lora in Fifth step.
```
bash merge_lora.sh
```
### Training the Knowledge Refinement Model
After training the Knowledge Refinement Model, you can start training the Knowledge Refinement Model.

(1) First step: Go to ``scripts`` folder, and start generating DPO sampling data based on the ``original train/dev dataset`` and the Generaton Model trained from the previous step.
```
cd scripts
bash kr_forward.sh
```

(2) Second step:
The data files generated in first step are merged using ``merge_response_file.py`` to obtain DPO sampling data. Then generate DPO train/dev dataset to train Knowledge Refinement Model based on these DPO samples.
```
bash get_kr_data.sh
```

(3) Third step: Use the DPO train/dev dataset from second step to train Knowledge Refinement Modul.
```
bash train_kr.sh
```

## Evaluating RAG-DDR
After training the RAG-DDR model, you can test the performance of RAG-DDR with the test data constructed above. You need to feed the test dataset that has retrieved the documents into the RAG-DDR model and go through the knowledge module and the generation module in turn to get the final answer.

(1) First step: You need to pass the ``original testing dataset`` into the Knowledge Refinement Module to filter out the retrieved external documents that are not relevant to the query and save the datastes with refinement documents in ``--output_path``. 
```
cd scripts
bash kr_inference.sh
```
(2) Second step: After getting the refinement dataset, you can feed it into the Generation Module to generate the responses and evaluate the effects of RAG-DDR.
```
bash gen_inference.sh
```
For different tasks, you need to set different task identifiers, evaluation metrics identifiers and generation max tokens for ``--task``, ``--metric`` and ``--max_new_token``:
| TASK | Task Identifiers | metrics identifiers |max tokens|
|------|----|------|-----|
| NQ |nq  | accuracy |32|
| TriviaQA  | tqa | accuracy | 32|
| MARCO QA | marco | rouge |100|
|  HotpotQA |hotpotqa | accuracy| 32| 
| T-REx | t-rex |accuracy | 32|
| WoW | wow | f1 | 32|

## Contact
If you have questions, suggestions, and bug reports, please email:
```
1837917467@qq.com  
```