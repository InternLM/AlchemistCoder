from mmengine.config import read_base

with read_base():
    from ..summarizers.groups.ds1000 import ds1000_summary_groups


code_passk_summary_groups = [
    # rename
    {'name': 'humaneval_pass@1(greedy)', 'subsets': [['openai_humaneval', 'humaneval_pass@1']]},
    {'name': 'humaneval_plus_pass@1(greedy)', 'subsets': [['humaneval_plus', 'humaneval_plus_pass@1']]},
    {'name': 'sanitized_mbpp_pass@1(greedy)', 'subsets': [['sanitized_mbpp', 'score']]},
    {'name': 'mbpp_plus_pass@1(greedy)', 'subsets': [['mbpp_plus', 'mbpp_plus_pass@1']]},
    {'name': 'humanevalx', 'subsets': ['humanevalx-python', 'humanevalx-cpp', 'humanevalx-go', 'humanevalx-java', 'humanevalx-js']},
    {'name': 'ds1000_pass@1(greedy)', 'subsets': [['ds1000', 'naive_average']]},
    # real add
]

summarizer = dict(
    dataset_abbrs=[
        # 'code',
        'humaneval_pass@1(greedy)',
        'humaneval_plus_pass@1(greedy)',

        'sanitized_mbpp_pass@1(greedy)',
        'mbpp_plus_pass@1(greedy)',

        'ds1000_pass@1(greedy)',
        ['ds1000_Pandas', 'accuracy'],
        ['ds1000_Numpy', 'accuracy'],
        ['ds1000_Tensorflow', 'accuracy'],
        ['ds1000_Scipy', 'accuracy'],
        ['ds1000_Sklearn', 'accuracy'],
        ['ds1000_Pytorch', 'accuracy'],
        ['ds1000_Matplotlib','accuracy'],

        'humanevalx',
        'humanevalx-python',
        'humanevalx-cpp',
        'humanevalx-go',
        'humanevalx-java',
        'humanevalx-js',

    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], [])
)
