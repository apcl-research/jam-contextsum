# Contextsum

- model with 1,024 block size and finetuned with 2.15m data is in ```/nublar/jam_1024/ckpt_350m_2.15mdata_1024.pt```
- model with 1,024 block size and finetuned with both 2.15m data and contextsum is in ```/nfs/dropbox/jam1024_2mcgpt_contextsumgpt4.pt```
- Data for continuing finetuning, refering to ```contextsum/data/method_holdout_context/context_continue/```
- Trained with all eyetracking data without any method holdout, refering to ```contextsum/data/method_holdout_context/context_all/``` (236 samples)
- New data with callgraph ```/nfs/projects/EyeContext/summaries/eyecontext_summaries_with_callgraph_new.pkl``` (394 samples)

## USE between context and summary
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0' python3 contextsum/use_with_context.py
```

## Train as-is
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2,3' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=2 train.py config/finetune_model_contextsum_method.py --outfilename=ckpt_context_finetuned_without_freeze.pt --always_save_checkpoint=True --gradient_accumulation_steps=16 --dataset=method_holdout_context/context_all
```
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=1 sample.py config/finetune_model_contextsum_method.py --outfilename=ckpt_context_finetuned_without_freeze.pt --prediction_filename=testset_without_freeze.txt --testdir=test_nosum/ --max_new_tokens=50 --dataset=method_holdout_context/context_all --num_samples=5 --temperature=0.5
```
## Continue finetuning with contextsum
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='2,3' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=2 train.py config/finetune_model_contextsum_method_continue.py --outfilename=ckpt_context_continue_finetuned.pt --always_save_checkpoint=True --gradient_accumulation_steps=16 --dataset=method_holdout_context/context_continue
```
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4222 --nnodes=1 --nproc_per_node=1 sample.py config/finetune_model_contextsum_method_continue.py --outfilename=ckpt_context_continue_finetuned.pt --prediction_filename=testset_continue.txt --testdir=test_nosum/ --max_new_tokens=50 --dataset=method_holdout_context/context_continue --num_samples=5 --temperature=0.5
```

## Freeze 75% of the parameters in the linear layer (freeze freeze freeze train)
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='0,1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4000 --nnodes=1 --nproc_per_node=2 train.py config/finetune_model_contextsum_method.py --outfilename=ckpt_context_finetuned_freeze.pt --always_save_checkpoint=True --gradient_accumulation_steps=16 --dataset=method_holdout_context/context_all --freeze=True
```
```
CUDA_DEVICE_ORDER='PCI_BUS_ID' CUDA_VISIBLE_DEVICES='1' OMP_NUM_THREADS=2 time torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:4000 --nnodes=1 --nproc_per_node=1 sample.py config/finetune_model_contextsum_method.py --outfilename=ckpt_context_finetuned_freeze.pt --prediction_filename=testset_freeze.txt --testdir=test_nosum/ --max_new_tokens=50 --dataset=method_holdout_context/context_all --num_samples=5 --temperature=0.5
```
<!---   -->
### Old stuff 
those are old scripts but those are still useful if we want to hold out one method each time and run a bash script to train each method holdout and compile data for each method holdout automatically
### Compile data for test and train
- compile data for method holdout  ```./compile_data_method.sh```
- compile data for method holdout with context  ```./compile_data_method_context.sh```

### Train and predict
- method holdout: ```./execute_method.sh```
- participant holdout: ```./execute_participant.sh```
