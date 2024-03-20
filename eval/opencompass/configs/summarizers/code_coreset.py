from mmengine.config import read_base

with read_base():
    from ..summarizers.groups.mmlu import mmlu_summary_groups
    from ..summarizers.groups.bbh import bbh_summary_groups

summarizer = dict(
    dataset_abbrs=[
        # '--------- 考试 Exam ---------',  # category
        'mmlu',
        # '--------- 推理 Reasoning ---------',  # category
        # '数学推理', # subcategory
        'gsm8k',
        # '综合推理', # subcategory
        "bbh",
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], []),
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore'),
)
