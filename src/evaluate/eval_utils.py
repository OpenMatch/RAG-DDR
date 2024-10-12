import json
import jsonlines
import ast
import re
import string
from collections import Counter
from rouge import Rouge
import torch
import torch.nn.functional as F

SPECIAL_TOKEN_LENGTH =64

def remove_after_tag(text_list, tag="</s>"):
    new_text_list = []
    for text in text_list:
        index = text.find(tag)
        if index != -1:
            text = text[:index + len(tag)]
        new_text_list.append(text)
    return new_text_list

def apply_softmax_to_tensor_list(tensor_list):
    softmax_list = []
    for tensor in tensor_list:
        softmax_tensor = F.softmax(tensor, dim=-1)
        softmax_list.append(softmax_tensor)
    return softmax_list

def find_max_positions(args,tensor_list):
    max_positions = []
    max_tensors = []
    for tensor in tensor_list:
        sub_max_positions=[]
        sub_max_tensors =[]
        tensor = tensor.view(int(tensor.size(0)/args.top_n),args.top_n,tensor.size(-2),tensor.size(-1))
        tensor  = tensor.sum(dim=1)
        for sub_tensor in tensor:
            max_idx = torch.argmax(sub_tensor)
            max_tensor = torch.max(sub_tensor)
            sub_max_positions.append(max_idx)
            sub_max_tensors.append(max_tensor)

        max_positions.append(sub_max_positions)
        max_tensors.append(sub_max_tensors)

    return max_tensors, max_positions

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data

def process_input_data(input_data, args,top_n,tokenizer):

    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]
            if "answer" in item:
                item["golds"] = item["answer"]
            if "possible_answers" in item:
                item["golds"] = ast.literal_eval(item["possible_answers"])
            if "answerKey" in item:
                item["golds"] = [item["answerKey"]]
            if "AnswerKey" in item:
                item["golds"] = [item["AnswerKey"]]

        if isinstance(item["golds"], str):
            item["golds"] = [item["golds"]]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        if "instruction" not in item and "input" in item:
            item["instruction"] = item["input"]

        if "instruction" not in item and "query" in item:
            item["instruction"] = item["query"]

        if args.task == "marco":
            item["instruction"] = 'Q: {}\nA:'.format(item["instruction"])

        if args.task == "tqa":
            item["instruction"] = 'Q: {}\nA:'.format(item["instruction"])

        if args.task == "nq":
            item["instruction"] = 'Q: {}\nA:'.format(item["instruction"])

        if args.task == "hotpotqa":
            item["instruction"] = 'Q: {}\nA:'.format(item["instruction"])

        if args.task == "musique":
            item["instruction"] = 'Q: {}\nA:'.format(item["instruction"])

        if args.task == 'wiki':
            item["instruction"] = 'Q: {}\nA:'.format(item["instruction"])

        if args.task == "fever":
            item[
                "instruction"] = "Is the following statement correct or not? Say true if it's correct; otherwise say false.\n\n### Input:\n {}\n\n### Response:\n".format(
                item["instruction"])

        if args.task == "aida":
            item[
                "instruction"] = "Output the Wikipedia page title of the entity mentioned between [START ENT] and [END ENT] in the given context\ncontext: {}\nAnswer:".format(
                item["instruction"])

        if args.task == "t-rex":
            item[
                "instruction"] = "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity. {}\nAnswer:".format(
                item["instruction"])

        if args.task == "eli5":
            item[
                "instruction"] = "Provide a paragraph-length response using simple words to answer the following question.\nQ: {}\nA:".format(
                item["instruction"])

        if args.task == "arc":
            item[
                "instruction"] = "Given four answer candidates, A, B, C and D, choose the best answer choice which can answer the following question.\n{}\nAnswer:".format(
                item["instruction"])

        if args.task == "wow":
            parts = item["instruction"].split('\n')
            formatted_items = ["Q: " + parts[i] + '\n' if i % 2 == 0 else "A: " + parts[i] + '\n' for i in
                               range(len(parts))]
            formatted_items = ''.join(formatted_items) + 'A: '
            item["instruction"] = formatted_items

        if args.task == "hellaswag":
            item[
                "instruction"] = "Given four answer candidates, A, B, C and D, choose the best answer choice which can finish the follow sentence.\n{} {}\nAnswer:".format(
                item["instruction"], item['candidate'])

        if args.task == "socialiqa":
            item[
                "instruction"] = "Given three answer candidates, A, B and C, choose the best answer choice which can answer the following question.\n{} {}\nAnswer:".format(
                item["instruction"], item['candidate'])

        if args.task == "piqa":
            item[
                "instruction"] = "Given two answer candidates, A and B, choose the best answer choice which can comply with the physical commonsense.\n{} {}\nAnswer:".format(
                item["instruction"], item['candidate'])

    if args.retrieval_augment:
        for item in input_data:
         
            passage_list = []

            if args.rerank:
                for psg in item['rerank_passage']:
                    if "text" in psg:
                        passage_list.append(psg['text'])
                    elif "passage_text" in psg:
                        passage_list.append(psg['passage_text'])
                    elif "segment" in psg:
                        passage_list.append(psg['segment'])               

            else:
                for psg in item['passage']:
                    if "text" in psg:
                        passage_list.append(psg['text'])
                    elif "passage_text" in psg:
                        passage_list.append(psg['passage_text'])
                    elif "segment" in psg:
                        passage_list.append(psg['segment'])

         
            passage_list = passage_list[:top_n]
            passage = '\n'.join(passage_list)
            

            token_query = tokenizer(item["instruction"])
            query_length = len(token_query.input_ids)
            token_aug_psg = tokenizer(passage)
            token_aug_psg = token_aug_psg.input_ids[:args.max_length-SPECIAL_TOKEN_LENGTH-query_length]
            new_passage = tokenizer.decode(token_aug_psg,skip_special_tokens=True)
            item["instruction"] = 'Background:\n' + new_passage + '\n\n' + item["instruction"]

    return input_data

def postprocess_output(pred):
    pred = pred.replace("</s>", "")

    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred


"""
The begin of Kilt data test code
"""


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def _acc_score(prediction, ground_truth):
    if ground_truth in prediction or ground_truth.lower() in prediction or ground_truth.capitalize() in prediction:
        return 1.0
    else:
        return 0.0


def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]


def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def test_kilt_em(task, metric, prediction, ground_truth):
    gold_candidate_answers = get_gold_answers(ground_truth)
    prediction = str(prediction).strip()
    if metric == 'em':
        score = _metric_max_over_ground_truths(
            task, _exact_match_score, prediction, gold_candidate_answers
        )

    elif metric == 'accuracy':
        score = _metric_max_over_ground_truths(
            task, _acc_score, prediction, gold_candidate_answers
        )
    elif metric == 'rouge':
        score = _metric_max_over_ground_truths(
            task, _rougel_score, prediction, gold_candidate_answers
        )

    elif metric == 'f1':
        score = _metric_max_over_ground_truths(
            task, _f1_score, prediction, gold_candidate_answers
        )
    return score


def _metric_max_over_ground_truths(task, metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if task == 'fever' and ground_truth in ["REFUTES", "SUPPORTS"]:
            ground_truth = "true" if ground_truth == "SUPPORTS" else "false"
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_gold_answers(gold):
    ground_truths = set()
    for item in gold["golds"]:
        if isinstance(item, str):
            ground_truths.add(item)
        elif "answer" in item and item["answer"] and len(item["answer"].strip()) > 0:
            ground_truths.add(item["answer"].strip())

    return ground_truths


"""
The end of Kilt data test code
"""


def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0




