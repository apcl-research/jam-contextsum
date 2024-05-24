#!/bin/bash
for i in {0..39}
do

	
	# method holdout with context (freeze 75% of the linear layer and embedding layer)
	cp /scratch/contextsum/pretrain/gemini/ckpt_pretrain.pt /scratch/contextsum/leave-one-out/gemini/ckpt_context_freeze_holdout_m$i.pt
	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2,3' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=2 train.py config/finetune_model_contextsum.py --outfilename=ckpt_context_freeze_holdout_m$i.pt --always_save_checkpoint=True --gradient_accumulation_steps=16 --dataset=method_holdout_context/context$i --freeze=True --out_dir=/scratch/contextsum/leave-one-out/gemini
	CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=1 sample.py config/finetune_model_contextsum.py --outfilename=ckpt_context_freeze_holdout_m$i.pt --prediction_filename=predict_context_freeze_holdout_m$i.txt --testdir=data/method_holdout_context/context$i/testset/ --max_new_tokens=50 --dataset=method_holdout_context/context$i --out_dir=/scratch/contextsum/leave-one-out/gemini
	sleep 5
	
done
