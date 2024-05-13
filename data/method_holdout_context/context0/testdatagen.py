# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from datasets import Dataset

import pickle
import random
import argparse
import bincomb
import os

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--funcom-file', type=str, default='/nfs/projects/EyeContext/summaries/eyecontext_summaries_with_calgraph.pkl')
    parser.add_argument('--holdout-method-index', type=str, default='0')
    parser.add_argument('--unique-methods-file', type=str, default='/nfs/projects/EyeContext/summaries/unique_methods.pkl')
    parser.add_argument('--data-dir', type=str, default='testset/m0/')
    parser.add_argument('--summary-dir', type=str, default='summary/')

    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    funcom_file = args.funcom_file
    holdout_method_index = args.holdout_method_index
    data_dir = args.data_dir
    summary_dir = args.summary_dir
    unique_methods_file = args.unique_methods_file

    funcom = pickle.load(open(funcom_file, 'rb'))
    all_unique_methods = pickle.load(open(unique_methods_file, 'rb'))
    holdout_method = all_unique_methods[int(holdout_method_index)]
    
    
    if (not os.path.exists(summary_dir)):
        os.mkdir(summary_dir)
    ref_summary = []
    for data in funcom:
        code = data['code'][0]
        if(code != holdout_method):
            continue
        summary = data['summary']
        summary = summary.split('. ')    

        presummary = summary[:len(summary) -1]     
        presummary = '. '.join(presummary)                                                               
        lastsummary = summary[-1]
        if(lastsummary == ""):
            lastsummary=summary[-1]
        ref_summary.append(lastsummary.strip())


    for id, summary in enumerate(ref_summary):
        file = open(f'{summary_dir}{id}.test', 'w')
        for index in range(0, len(ref_summary)):
            file.write(f'{index}<SEP>{ref_summary[id]}\n' )
        file.close()




