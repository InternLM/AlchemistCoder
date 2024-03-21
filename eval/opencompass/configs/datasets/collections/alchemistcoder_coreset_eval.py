from mmengine.config import read_base

with read_base():
    from ..alchemistcoder_coreset_eval_datasets.mmlu.mmlu_gen_a484b3 import mmlu_datasets
    from ..alchemistcoder_coreset_eval_datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from ..alchemistcoder_coreset_eval_datasets.bbh.bbh_gen_5b92b0 import bbh_datasets

code_coreset_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])
