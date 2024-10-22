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

        if args.task == "t-rex":
            item[
                "instruction"] = "Given the input format 'Subject Entity [SEP] Relationship Type,' predict the target entity. {}\nAnswer:".format(
                item["instruction"])

        if args.task == "wow":
            parts = item["instruction"].split('\n')
            formatted_items = ["Q: " + parts[i] + '\n' if i % 2 == 0 else "A: " + parts[i] + '\n' for i in
                               range(len(parts))]
            formatted_items = ''.join(formatted_items) + 'A: '
            item["instruction"] = formatted_items

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


def test_kilt(task, metric, prediction, ground_truth):
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






