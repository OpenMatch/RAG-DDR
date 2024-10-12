from vllm import LLM, SamplingParams
import argparse
import json
from tqdm import  tqdm
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os



# Prompt used here.
minicpm_prompt_template = "<用户>{}<AI>"

augment_templeta = 'Background:\n{}\n\n{}'


multi_choice = 'The following is multiple choice question. Please choose the best answer choice which can answer the following question.\n{}\nAnswer:'
QA_templeta = 'Q: {}\nA:'

Mult_COT_templeta = "Please answer multiple choice question and choose the best answer choice first. Then give your explanation between [<COT] and [COT>]."
QA_COT_templeta = "Please answer the question. Then give your explanation between [<COT] and [COT>]."

COT_few_shot = 'question: {}\nAnswer:{}\n[<COT] {} [COT>]\n '
COT_question = 'question: {}\nAnswer:'


aqua_rat_shot = [{"question": "If six persons sit in a row, then the probability that three particular persons are always together is ?\\nOptions:\\n(A) 1/6\\n(B) 1/0\\n(C) 1/5\\n(D) 1/4\\n(E) 1/1", "answer": "(C)",
                  "cot":"Six persons can be arranged in a row in 6! ways. Treat the three persons to sit together as one unit then there four persons and they can be arranged in 4! ways. Again three persons can be arranged among them selves in 3! ways. Favourable outcomes = 3!4! Required probability = 3!4!/6! = 1/5"},
                 {"question": "From the set of numbers x, y, t, z, s and w, how many different combinations can we have without the t in them? Ex:. (x,y), (x), (w,z,y,x,s), etc and (x,y)=(y,x)\\nOptions:\\n(A) 10\\n(B) 14\\n(C) 15\\n(D) 16\\n(E) 31", "answer": "(E)",
                  "cot":"Another way: Any letter (x, y, z, w, s) can be included or not. So, we have 2^5 combinations - 1 empty combination = 31 combinations"},
                 {"question": "The total number of problems in the mathematics and algebra books is 28 more than the total number of problems in the algebra and geometry books. How much lesser is the number of problems in the geometry book as compared to the mathematics book?\\nOptions:\\n(A) 30\\n(B) 18\\n(C) 28\\n(D) 25\\n(E) 32","answer":"(C)",
                  "cot":"(Mathematics + Algebra) - (Algebra + Geometry) = 28\\nMathematics - Geometry = 28"}]
                  
ecqa_shot = [{"question": "The aggressive soldiers were getting drunk, what are they at they at risk of doing with the locals?\\nOptions:\\n- nausea\\n- erections\\n- fights\\n- relaxation\\n- procreation", "answer": "fights",
                 "cot": "Aggressiveness means ready or likely to attack or confront. Getting drunk encourages aggressiveness. Attacking and confronting comes under fights."},
            {"question": "What does exercising immediately lead to?\\nOptions:\\n- relaxation\\n- exhaustion\\n- energetic\\n- become stronger\\n- use energy", "answer": "exhaustion",
             "cot": "Exhaustion is caused by exercising. Exercising immediately lead to exhaustion."},
            {"question": "The blood was everywhere in the operating room, what were the surgeons operating on?\\nOptions:\\n- soccer game\\n- musle\\n- vein\\n- animals\\n- person", "answer": "person",
            "cot": "Operating room is a place. Surgeries are performed in the operating room. Surgeries are performed on human beings. Operating rooms are found in hospitals for human beings."}
            ]

gsm8k_shot = [{"question": "There are 12 carpets in house 1, 20 carpets in house 2, and 10 carpets in house 3. If house 4 has twice as many carpets as house 3, how many carpets do all 4 houses have in total?", "answer": "62",
               "cot": "House 4 has 2 * 10 house 3 carpets = 20 carpets. The total number of carpets across all houses is 12 + 20 + 10 + 20 = 62."},
              {"question": "CJ, KJ, and AJ collect stamps. CJ has 5 more than twice the number of stamps that KJ has, and KJ has half as many as AJ. If the three boys have 930 stamps all together, how many stamps does AJ have?", "answer": "370",
               "cot": "Let x represent the number of stamps for AJ. KJ:x / 2 stamps. CJ:5 + 2(x / 2) = 5 + x. Total:x + (x / 2) + 5 + x = 930. (5 / 2)x + 5 = 930. (5 / 2)x = 925. x = 925(2 / 5) = 370 stamps."},
              {"question": "Lily types 15 words a minute and takes a 2-minute break every 10 minutes. How long does it take for Lily to type 255 words?", "answer": "19",
              "cot": "It would take Lily 255 words / 15 wpm = 17 minutes to type without taking a break. Since Lily takes a break after 10 minutes of typing she takes 17 minutes + 2 minutes = 19 minutes."}]

strategyqa_shot = [{"question": "Is radioactive waste a plot device for many shows?", "answer": "yes",
                    "cot": "Radioactive isotopes in an ooze-like waste cause turtles to become the Teenage Mutant Ninja Turtles. In the Fox animated hit, Family Guy, radioactive waste is used to turn give the main characters superpowers. The superhero 'Daredevil' encounters radioactive waste that blinds him as a child and gives him super powers."},
                   {"question": "Will a celibate cleric likely suffer a stoning in Somalia?", "answer": "no",
                    "cot": "A cleric is the term for a Muslim priest. Celibate people remain chaste and do not engage in relations with others. Stoning is a penalty in Somalia used to punish adulterers. Many Islamic militants have been in control of various parts of Somalia."},
                   {"question": "Is Bill Gates the wealthiest of the Baby Boomers?", "answer": "no",
                    "cot": "The Baby Boomers are the generation born between the years 1946-1964. Bill Gates was born on October 28, 1955 and has a net worth of 108 billion as of 2020. Jeff Bezos was born on January 12, 1964 and has a net worth of 160 billion as of 2020."}
                   ]
read_type = ['coqa']


llama_multi_choice = 'Please answer the multiple choice questions below and output only the choice.\n{}\nAnswer:'
llama_QA_COT_templeta = "Please answer the question and only output the answer. Then give your explanation between [<COT] and [COT>]."

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            item['id'] = idx
            data.append(item)
    return data

def split_list_evenly(lst, n):
    length = len(lst)
    avg = length // n
    remainder = length % n
    out = []
    start = 0

    for i in range(n):
        end = start + avg + (1 if i < remainder else 0)
        out.append(lst[start:end])
        start = end
    
    return out  

      
class LLMDataset(Dataset):
    def __init__(self,data_list,args,tokenizer):
        self.data_list = data_list
        self.args = args
        self.tokenizer = tokenizer
        
    def process_query_with_psg(self, item):

        if self.args.llama_style:
            if item['data_type'] in ['commonsense_qa', 'math_qa',"aqua_rat","ecqa"]:
                raw_input = llama_multi_choice.format(item['question'])

            if item['data_type'] == 'coqa' or item['data_type'] =='web_questions' or item['data_type'] =='wiki_qa' or item['data_type'] =='yahoo_answers_qa' \
            or item['data_type'] =="marcoqa":
                raw_input = QA_templeta.format(item['question'])
                
            if item['data_type'] in ["gsm8k","strategyqa"]:
                
                if item['data_type'] == "gsm8k":
                    shot_example = gsm8k_shot
                    
                if item['data_type'] =="strategyqa":
                    shot_example = strategyqa_shot
                    
                shot_prompt =""
                for few_shot in shot_example:
                    shot_prompt = shot_prompt+COT_few_shot.format(few_shot["question"],few_shot["answer"],few_shot["cot"])
                raw_input = llama_QA_COT_templeta + shot_prompt + COT_question.format(item['question']) 

        else:
            if item['data_type'] == 'commonsense_qa' or item['data_type'] == 'math_qa':
                raw_input = multi_choice.format(item['question'])
            
            if item['data_type'] == 'coqa' or item['data_type'] =='web_questions' or item['data_type'] =='wiki_qa' or item['data_type'] =='yahoo_answers_qa' \
            or item['data_type'] =="marcoqa":
                raw_input = QA_templeta.format(item['question'])
                
                
            if item['data_type'] in ["aqua_rat","ecqa"]:
                
                if item['data_type'] == "aqua_rat":
                    shot_example = aqua_rat_shot
                
                if item['data_type'] =="ecqa":
                    shot_example = ecqa_shot
                                
                shot_prompt =""
                for few_shot in shot_example:
                    shot_prompt = shot_prompt+COT_few_shot.format(few_shot["question"],few_shot["answer"],few_shot["cot"])
                raw_input = Mult_COT_templeta + shot_prompt + COT_question.format(item['question']) 
                
            if item['data_type'] in ["gsm8k","strategyqa"]:
                
                if item['data_type'] == "gsm8k":
                    shot_example = gsm8k_shot
                    
                if item['data_type'] =="strategyqa":
                    shot_example = strategyqa_shot
                    
                shot_prompt =""
                for few_shot in shot_example:
                    shot_prompt = shot_prompt+COT_few_shot.format(few_shot["question"],few_shot["answer"],few_shot["cot"])
                raw_input = QA_COT_templeta + shot_prompt + COT_question.format(item['question']) 
        
        if self.args.llama_style:

            if self.args.chat_templete:
                item['raw_input'] = [ {"role": "user", "content": raw_input},]
                item['raw_input'] = self.tokenizer.apply_chat_template(item['raw_input'], add_generation_prompt=True, tokenize=False)
            
            else:
                item['raw_input'] = raw_input

        
        else:
            item['raw_input'] = minicpm_prompt_template.format(raw_input)
        
        if item['data_type'] in read_type:
            item = self.get_one_psg(item,raw_input)
        else:
            item = self.get_more_psg(item,raw_input,)

            
        return item
    
    
    def get_one_psg(self, item,raw_input):
        item['augment_input']=[]
        item['augment_type'] = []
        
        psg_list = []
        psgs = item['passage']
        for p in psgs:
            psg_list.append(p['segment'])
            
        for idx, psg in enumerate(psg_list):
            item['augment_input'].append(minicpm_prompt_template.format(
                augment_templeta.format(psg, raw_input)))
            
            item['augment_type'].append('aug_'+ str(idx+1))
        
        return item

    
    def get_more_psg(self,item,raw_input):
        item['augment_input']=[]
        item['augment_type'] = []

        psg_list = []

        if len(item['rerank_passage'])>=5:
            #item['rerank_passage'] represents the passage after we use knowledge refine
            psgs = item['rerank_passage'][:5]
        else:
            psgs = item['rerank_passage']+item['passage'][:5-len(item['rerank_passage'])]

        token_query = self.tokenizer([raw_input])
        query_length = len(token_query.input_ids[0])

        for p in psgs:
             psg_list.append(p['segment'])
        
        #psg_list[:1] means we only use the top1 psg
        for idx, psg in enumerate(psg_list[:1]):

            token_aug_psg = self.tokenizer([psg])
            token_aug_psg = token_aug_psg.input_ids[0][:4096-64-query_length]
            new_psg = self.tokenizer.decode(token_aug_psg,skip_special_tokens=True)

            if self.args.llama_style:

                if self.args.chat_templete:
                    aug_input = [{"role": "user", "content": augment_templeta.format(new_psg, raw_input)}]
                    aug_input = self.tokenizer.apply_chat_template(aug_input, add_generation_prompt=True, tokenize=False)
                else:
                    aug_input = raw_input
                item['augment_input'].append(aug_input)           
            else:
                item['augment_input'].append(minicpm_prompt_template.format(
                augment_templeta.format(new_psg, raw_input)))                 
            item['augment_type'].append('aug_'+ str(idx+1))


        aug_psg = '\n'.join(psg_list[:5])
        token_aug_psg = self.tokenizer([aug_psg ])
        token_aug_psg = token_aug_psg.input_ids[0][:4096-64-query_length]
        new_psg = self.tokenizer.decode(token_aug_psg,skip_special_tokens=True)

        if self.args.llama_style:

            if self.args.chat_templete:
                aug_input = [{"role": "user", "content": augment_templeta.format(new_psg, raw_input)}]
                aug_input = self.tokenizer.apply_chat_template(aug_input, add_generation_prompt=True, tokenize=False)
            else:
                aug_input = raw_input
            item['augment_input'].append(aug_input)
        
        else:
            item['augment_input'].append(minicpm_prompt_template.format(
            augment_templeta.format(new_psg, raw_input)))
        item['augment_type'].append('aug_1-5')
            
        return item
            
    def __getitem__(self, index):
        item = self.data_list[index]       
        item = self.process_query_with_psg(item)

        if index == 0:
            print(item)
       
        return item
    
    def __len__(self):
        return len(self.data_list)
    
    def Collactor (self, batch):
        
        raw_input = [f['raw_input'] for f in batch]
        id = [f['id'] for f in batch]
        answers = [f['answer'] for f in batch]
        augment_input = [f['augment_input'] for f in batch]
        type = [f['augment_type'] for f in batch]
        data_type = [f['data_type'] for f in batch]
        
        return{'id':id,
               'raw_input':raw_input,
               'answers':answers,
               'augment_input':augment_input,
               'type': type,
               'data_type':data_type,
               }
        
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str,
                        default=None,
                        help="The path of the training/evaluation data file to be processed.",
                        )
    parser.add_argument('--model_name_or_path', type=str,
                        default=None,)
    parser.add_argument('--output_path', type=str,
                        default=None,)
    parser.add_argument('--batch_size', type=int,
                        default=8)
    parser.add_argument('--chat_templete', action='store_true',
                        help="Template used by chat model.",
                        )
    parser.add_argument('--llama_style', action='store_true',
                        help="Template used by llama model.",)
    parser.add_argument('--loop', type=int,
                        default=5,
                        help="The number of multiple sampling rounds.",
                        )
    parser.add_argument('--cut_chunk', type=int,
                        default=8,
                        help="The number of data segments.",
                        )
    parser.add_argument('--number_chunk', type=int,
                        default=0,
                        help="The current index of data segments.",
                        )
    
    args = parser.parse_args()
    
    
    print("The parameter configuration is as follows:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    input = read_jsonl(args.input_data_path)
    
    split_input = split_list_evenly(input,args.cut_chunk)
    input_list = split_input[args.number_chunk]
    
    print("----------------case-----------------------")
    print(input_list[0]['question'])
    print("----------------begin----------------------")


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = LLMDataset(input_list,args,tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn= dataset.Collactor,)
    
    temperature_list = [0.5,0.6,0.7,0.8,0.9]
    all_save_list = [[] for _ in range(len(input_list))]
    
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size= 1, trust_remote_code=True, dtype='bfloat16',)
    for temp in tqdm(temperature_list):
        params_dict = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "temperature": temp,
            "top_p": 0.8,
            "top_k": -1,
            "use_beam_search": False,
            "length_penalty": 1,
            "early_stopping": False,
            "stop": None,
            "stop_token_ids": None,
            "ignore_eos": False,
            "max_tokens": 100,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }

        # Create a sampling params object.
        sampling_params = SamplingParams(**params_dict)

        # Create an LLM.
        locate = 0
        prior = 0
        next = 0


        for batch in tqdm(dataloader):

            batch_raw_query = batch['raw_input']
            batch_augment_query = batch['augment_input']
            batch_augment_query = sum(batch_augment_query ,[])
            answers =batch['answers']
            ids = batch['id']
            type_list = batch['type']
            data_type_list = batch['data_type']
            # Both locate and prior are reset
            locate = next
            prior = next
                      
            for loop in range(args.loop):
                # Each time a large loop starts, locate is reset to prior
                locate = prior
                
                raw_outputs = llm.generate(batch_raw_query, sampling_params)
                augment_outputs = llm.generate(batch_augment_query, sampling_params)
                augment_outputs = [augment_outputs[i:i + 2] for i in range(0, len(augment_outputs), 2)]
                
                for id, raw, aug, an, type,data_type in zip(ids,raw_outputs, augment_outputs, answers, type_list,data_type_list):
                    
                    # Only the first large loop needs to put data, so here loop=0
                    if loop == 0 and temp == 0.5:
                        all_save_list[locate] = {}
                        all_save_list[locate]['id'] = id
                        all_save_list[locate]['context']=[]
                    raw_text = raw.outputs[0].text
                    tempt_1 = {}
                    tempt_1['text'] = raw_text
                    tempt_1['temperature'] = temp
                    tempt_1['type'] = 'raw'
                    tempt_1['data_type']=data_type
                    tempt_1['loop'] = str(loop+1)
                    all_save_list[locate]['context'].append(tempt_1)
                    
                    aug_text_list =[]
                    for aug_text in aug:
                        aug_text_list.append(aug_text.outputs[0].text)
                        
                    for idx, aug_text in enumerate(aug_text_list):
                        tempt_2 = {}
                        tempt_2['text'] = aug_text
                        tempt_2['temperature'] = temp
                        tempt_2['type'] = type[idx]
                        tempt_2['data_type']=data_type
                        tempt_2['loop'] = str(loop+1)
                        all_save_list[locate]['context'].append(tempt_2)
                                           
                    locate+=1
                    if loop == 0:
                        # next is the next starting index
                        next = locate
                    
    out_put_path = os.path.join(args.output_path,'all_text_data.{}.jsonl'.format(args.number_chunk))
    with open(out_put_path, 'w', encoding='utf-8') as f:
        for item in all_save_list:
            f.write(json.dumps(item) + '\n')

    print("---------finish--------------")

if __name__ == "__main__":
    main()


