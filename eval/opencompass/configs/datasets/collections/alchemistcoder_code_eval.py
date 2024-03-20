
from mmengine.config import read_base

with read_base():
    from ..alchemistcoder_code_eval_datasets.humaneval_gen_8e312c import humaneval_datasets
    from ..alchemistcoder_code_eval_datasets.humaneval_plus_gen_8e312c import humaneval_plus_datasets
    from ..alchemistcoder_code_eval_datasets.sanitized_mbpp_gen_1e1056 import sanitized_mbpp_datasets
    from ..alchemistcoder_code_eval_datasets.humanevalx_gen_620cfa import humanevalx_datasets
    from ..alchemistcoder_code_eval_datasets.ds1000_compl_service_eval_gen_cbc84f import ds1000_datasets
    from ..alchemistcoder_code_eval_datasets.mbpp_plus_gen_94815c import mbpp_plus_datasets

compassbench_v1_code_datasets = sum([v for k, v in locals().items() if (k.endswith("_datasets") and not k.endswith("_passk_datasets") and not k.endswith("_repeat10_datasets"))], [])
