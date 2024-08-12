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
import math

random.seed(1337)

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num-proc', type=int, default=4)
    parser.add_argument('--funcom', type=str, default='/scratch/chiayi/contextsum/contextsum.pkl')
    parser.add_argument('--unique-methods', type=str, default='/nfs/projects/EyeContext/summaries/unique_methods.pkl')
    parser.add_argument('--holdout-method-index', type=str, default='0')
    parser.add_argument('--data-dir', type=str, default='bins/')
    parser.add_argument('--test-data-dir', type=str, default='testset/')
    args = parser.parse_args()

    num_proc = outdir = args.num_proc
    funcom = args.funcom
    holdout_method_index = args.holdout_method_index
    data_dir = args.data_dir
    test_data_dir = args.test_data_dir
    unique_methods_file = args.unique_methods
    
    if (not os.path.exists(test_data_dir)):
        os.mkdir(test_data_dir)

    unique_methods = pickle.load(open(unique_methods_file, "rb"))

    alldata = pickle.load(open(funcom, 'rb'))
    holdout_method = unique_methods[int(holdout_method_index)]
    traindata = []
    testdata = []
    valdata = []
    
    random.shuffle(alldata)
    for data in alldata:
        method = data['code'][0]
        if(method == holdout_method):
            testdata.append(data)
            continue
        traindata.append(data)
    
    valdata = traindata[:5]
    count_test_samples = 0
    for dat in testdata[:]:
        with open(f'{test_data_dir}{count_test_samples}.txt', 'w') as f:
            code = dat['code'][0]
            contextsum = dat['jam_contextsum']
            prompt = f'TDAT\n{code}\nCONTEXT\n{contextsum}\nSUMMARY\n'
            f.write(prompt)
            count_test_samples += 1
    random.shuffle(alldata)

    
    
    pt = int(len(alldata) * 1.0)
    count_train = 0 
    count_val = 0
    count_test = 0
    for partnum in range(0, 1):

        print(f'starting part {partnum}')

        txtfiles = list()
        txtfiles_val = list()
        bin_file_path = data_dir + f'/val_2pt_p{partnum}.bin'

        if os.path.isfile(bin_file_path):
            continue

        start_pt = (partnum * pt)
        end_pt = ((partnum+1) * pt)

        fundats_2pt_px = alldata[start_pt:end_pt]
        for data in tqdm(fundats_2pt_px):
            ori_code = data['code'][0]
            contextsum = data['jam_contextsum']
            summary = data['summary']
            summary = summary.split('.')
            if(summary[-1] != ''):
                summary = summary[-1].strip()
            else:
                summary = summary[-2].strip()
            if data in valdata:
                with open(f'tmp/{count_val}_val', 'w') as f:
                    prompt = f'TDAT\n{ori_code}\nCONTEXT\n{contextsum}\nSUMMARY\n{summary}'
                    f.write(prompt)
                txtfiles_val.append(f'tmp/{count_val}_val')
                count_val += 1
            elif data in testdata:
                count_test += 1
                continue
            else:
                with open(f'tmp/{count_train}_train', 'w') as f:
                    prompt = f'TDAT\n{ori_code}\nCONTEXT\n{contextsum}\nSUMMARY\n{summary}'
                    f.write(prompt)
                txtfiles.append(f'tmp/{count_train}_train')
                count_train += 1
        if( txtfiles == [] and txtfiles_val ==[]):
            continue
        elif(txtfiles_val ==[]):
            dataset = load_dataset('text', data_files={'train': txtfiles}, sample_by="document")
        elif(txtfiles ==[]):
            dataset = load_dataset('text', data_files={'val':txtfiles_val}, sample_by="document")
        else: 
            dataset = load_dataset('text', data_files={'train': txtfiles, 'val':txtfiles_val}, sample_by="document")
        shmdir = 'tmp/'

        pickle.dump(dataset, open(f'pkls/dataset_funcom_2pt_p{partnum}.pkl', 'wb'))



        # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
        enc = tiktoken.get_encoding("gpt2")
        def process(example):
            ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
            ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the dataset
        tokenized = dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = os.path.join(data_dir, f'{split}_2pt_p{partnum}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

            print(f"writing {filename}...")
            idx = 0
            for example in tqdm(dset):
                arr[idx : idx + example['len']] = example['ids']
                idx += example['len']
            arr.flush()
    print(f'number of training samples: {count_train}')
    print(f'number of val samples: {count_val}')
    print(f'number of test samples: {count_test}')
    bincomb.main('bins/')
