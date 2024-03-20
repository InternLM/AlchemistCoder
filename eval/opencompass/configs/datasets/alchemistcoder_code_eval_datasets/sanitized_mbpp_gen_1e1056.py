from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SanitizedMBPPDataset, MBPPEvaluator

sanitized_mbpp_reader_cfg = dict(
    input_columns=['text', 'test_list'], output_column='test_list_2')

sanitized_mbpp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                role='HUMAN',
                prompt= "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n\nPlease refer the given examples and generate a python function for my problem. \nExamples are listed as follows:\n\nQuestion: \nYou are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should satisfy the following assertion:\n```python\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \n assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n```\n\nYour code should start with a [BEGIN] tag and end with a [DONE] tag.\n \nAnswer:\n[BEGIN]\ndef similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)\n[DONE] \n\n \n Question: \nYou are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should satisfy the following assertion:\n```python\n assert is_not_prime(2) == False \n assert is_not_prime(10) == True \n assert is_not_prime(35) == True \n```\n\nYour code should start with a [BEGIN] tag and end with a [DONE] tag.\n \nAnswer:\n[BEGIN]\nimport math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result\n[DONE] \n\n \n Question: \nYou are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should satisfy the following assertion:\n```python\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n```\n\nYour code should start with a [BEGIN] tag and end with a [DONE] tag.\n\nAnswer:\n[BEGIN]\nimport heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums\n[DONE] \n\n\nHere is my problem:\n\nQuestion: \nYou are an expert Python programmer, and here is your task: {text}\nYour code should satisfy the following assertion:\n```python\n {test_list}  \n```\n\nYour code should start with a [BEGIN] tag and end with a [DONE] tag.\n\nAnswer:\n"),          
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

sanitized_mbpp_eval_cfg = dict(evaluator=dict(type=MBPPEvaluator), pred_role="BOT")

sanitized_mbpp_datasets = [
    dict(
        type=SanitizedMBPPDataset,
        abbr='sanitized_mbpp',
        path='./data/mbpp/sanitized-mbpp.jsonl',
        reader_cfg=sanitized_mbpp_reader_cfg,
        infer_cfg=sanitized_mbpp_infer_cfg,
        eval_cfg=sanitized_mbpp_eval_cfg)
]
