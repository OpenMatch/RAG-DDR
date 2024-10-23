# RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards

Source code for our paper :  
[RAG-DDR: Optimizing Retrieval-Augmented Generation Using Differentiable Data Rewards](https://arxiv.org/abs/2410.13509)

## Requirement
**1. Install the following packages using Pip or Conda under this environment**

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
### Prepare the training and test data**
* Use `git clone` to download this project:
```bash
git clone https://github.com/OpenMatch/RAG-DDR.git
cd RAG-DDR
```
* Construct ``original train/dev dataset``<br>
** In order to construct the train/dev dataset for RAG-DDR, you can follow Table 5 in the paper to collect the corresponding dataset and process the data into the following jsonl format. Similarly, you can collect more datasets on your own and process them into the jsonl format below for subsequent DDR training.

```
{'id': str # The id of the data, you need to reassign a new id to the data you collected, with values from '0' to 'n-1' and 'n' being the total number of data.
 'question': str # The question of data.
 'answer': str # The ground truth of data.
 'data_type': str # Which task this data belongs to.
 'cot': str/None # Does this data have a labeled COT, if so it is provided, if not it is None.
 }
```

** After that, you can use bge-large to retrieve relevant documents from MS MARCO 2.0 for the above constructed data. The MS MARCO 2.0 corpus can be downloaded from here. specifically，you can go to the ``data`` folder to retrieve the relevant documents for the corresponding data. You need to use bge to encode all the passages as embedding and saved in ``--output_dir``:

```
cd data
bash getembedding.sh
```
**Then, you need to get the appropriate trec file for the original train/dev dataset which are saved in ``--query_path`` and save it in ``--trec_save_path``:

```
bash retrieve.sh
```
**Finally, you can retrieve the relevant passages for each piece of data based on the ``--trec_save_path`` and save the ``original train/dev dataset`` with the retrieved passages in  ``--output_path``:

```
bash construct_psg_data.sh
```
* Construct ``original testing dataset``<br>
For constructing the ``original testing dataset``, you can download KILT's test dataset from here and select the corresponding test dataset according to Table 5 in the paper. After that, you can retrieve the relevant passages for each test dataset to get ``original testing dataset`` based on the above data processing steps.

* Download the constructed data directly<br>
We also provide the constructed ``original train/dev dataset`` and ``original testing dataset`` that will be released subsequently.

### Training the Generation Model
**After constructing the training and test data, you can start training the RAG-DDR model. You need to train the Generation Model in RAG-DDR by constructing DPO train/dev dataset with the previously constructed original train/dev dataset.** 

* First step: You first need to download Minicpm-2.4B-sft and lama3-8B-Instruct model.

* Second step: Go to ``scripts`` folder, and start generating sample data using ``original train/dev dataset``:
```
cd scripts
bash generation_forward.sh
```
* Third step: The data files generated in the second step are merged using ``merge_response_file.py`` to obtain DPO samples. Then generate DPO train/dev dataset based on these DPO samples:

```
bash get_generation_data.sh
```
* Fourth step: Use the DPO train/dev dataset to train Generation Modul, you can choose Minicpm-2.4B-sft or lama3-8B-Instruct model as Generation Modul:
```
bash train.sh
```
* Fifth step: Combine the weights of the Generation Modul trained using lora in Fourth step:
```
bash train.sh
```

### Evaluating RAG-DDR
**After training the RAG-DDR model, you can test the performance of RAG-DDR with the test data constructed above. You need to feed the test data that has retrieved the passages into the RAG-DDR model and go through the knowledge module and the generation module in turn to get the final answer.**

* First step: You need to pass the ``original testing dataset`` into the Knowledge Refinement Module to filter out the retrieved external documents that are not relevant to the query and save the datastes with refinement documents in ``--output_path``.
```
cd scripts
bash kr_inference.sh
```
* Second step: After getting the refinement dataset, you can feed it into the Generation Module to generate the responses and evaluate the effects of RAG-DDR. For different tasks, you need to assign different task identifiers and evaluation metrics identifiers for ``--task`` and ``--metric``. Specifically, the correspondence between tasks, task identifiers and metric identifiers is:

| TASK | Task Identifiers | metrics identifiers |
|------|----|------|
| NQ |nq  | accuracy |  
| TriviaQA  | tqa | accuracy | 
| MARCO QA | marco | rouge | 
|  HotpotQA |hotpotqa | accuracy|  
| T-REx | t-rex |accuracy | 
| WoW | wow | f1 | 

```
cd scripts
bash gen_inference.sh
```