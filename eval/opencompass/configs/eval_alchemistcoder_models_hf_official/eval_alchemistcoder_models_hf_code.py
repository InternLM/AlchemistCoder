import os.path as osp
from copy import deepcopy

from mmengine.config import read_base

from opencompass.models import HuggingFaceCausalLM

with read_base():
    # datasets
    from ..datasets.collections.alchemistcoder_code_eval import compassbench_v1_code_datasets
    # summarizer
    from ..summarizers.code_summarizers import summarizer
    # clusters
    from ...clusters.aliyun_llmit import infer, eval
    # lark robot
    from ...lark import lark_bot_url

# ------------------ change here ↓ ------------------
infer['runner']['max_num_workers'] = 128
eval['runner']['max_num_workers'] = 128
eval['partitioner']['n'] = 1
# ------------------ change end ------------------

meta_template = dict(
    begin="""""",
    round=[
        dict(role='HUMAN', begin='<|Human|>െ', end='\n '),
        dict(role='BOT', begin='<|Assistant|>െ', end='ി\n ', generate=True),
    ],
    eos_token_id=30387)


meta_template_ds = dict(
    begin="""""",
    round=[
        dict(role='HUMAN', begin='<|User|>::', end='\n '),
        dict(role='BOT', begin='<|Assistant|>::', end='<|EOT|>\n ', generate=True),
    ],
    eos_token_id=32021)



# ------------------ default settings ↓ ------------------
# careful to change the following settings
AlchemistCoder_L_7B_hf_path = 'your AlchemistCoder-L-7B path'
AlchemistCoder_L_7B_hf = dict(
    type=HuggingFaceCausalLM,
    abbr='AlchemistCoder_L_7B_hf',
    path=AlchemistCoder_L_7B_hf_path,
    tokenizer_path=AlchemistCoder_L_7B_hf_path,
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True
    ),
    # tokenizer_kwargs=dict(
    #     padding_side='left',
    #     truncation_side='left',
    #     trust_remote_code=True,
    #     use_fast=False,),
    max_out_len=1024,
    max_seq_len=8192,
    batch_size=1,
    meta_template=meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
    end_str="ി",
)

AlchemistCoder_CL_7B_hf_path = 'your AlchemistCoder-CL-7B path'
AlchemistCoder_CL_7B_hf = dict(
    type=HuggingFaceCausalLM,
    abbr='AlchemistCoder_CL_7B_hf',
    path=AlchemistCoder_CL_7B_hf_path,
    tokenizer_path=AlchemistCoder_CL_7B_hf_path,
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True
    ),
    # tokenizer_kwargs=dict(
    #     padding_side='left',
    #     truncation_side='left',
    #     trust_remote_code=True,
    #     use_fast=False,),
    max_out_len=1024,
    max_seq_len=8192,
    batch_size=1,
    meta_template=meta_template,
    run_cfg=dict(num_gpus=1, num_procs=1),
    end_str="ി",
)

AlchemistCoder_DS_7B_hf_path = 'your AlchemistCoder-DS-6.7B path'
AlchemistCoder_DS_7B_hf = dict(
    type=HuggingFaceCausalLM,
    abbr='AlchemistCoder_DS_6.7B_hf',
    path=AlchemistCoder_DS_7B_hf_path,
    tokenizer_path=AlchemistCoder_DS_7B_hf_path,
    model_kwargs=dict(
        device_map='auto',
        trust_remote_code=True
    ),
    # tokenizer_kwargs=dict(
    #     padding_side='left',
    #     truncation_side='left',
    #     trust_remote_code=True,
    #     use_fast=False,),
    max_out_len=1024,
    max_seq_len=8192,
    batch_size=1,
    # meta_template=meta_template_ds,
    run_cfg=dict(num_gpus=1, num_procs=1),
)

models = [
    AlchemistCoder_L_7B_hf,
    AlchemistCoder_CL_7B_hf,
    AlchemistCoder_DS_7B_hf,
]

# set all models
model_dataset_combinations = []
datasets = []

# base
model_dataset_combinations.append(dict(models=models, datasets=compassbench_v1_code_datasets))

datasets.extend(compassbench_v1_code_datasets)

# ------------------ default settings end ------------------