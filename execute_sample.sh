#!/bin/bash

## human study data


## large testset

CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4111 --nnodes=1 --nproc_per_node=1 sample.py config/finetune_model_contextsum_method.py --outfilename=jam1024_2mcgpt_contextsumgpt4_longtest_gemini.pt --prediction_filename=predict_large_testset_gemini.txt --testdir=test_nosum/ --max_new_tokens=50 --dataset=./ --out_dir=/scratch/contextsum/pretrain/gemini/



