import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-jam-context'
eval_interval = 5
eval_iters = 5
wandb_log = False 
wandb_project = 'jam-context'
wandb_run_name = 'jam-context'

dataset = 'method_holdout_context'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = True 

dropout = 0.2


# jam-cgpt 2.15m has 403,770,021 tokens

# model iters
# 38m parameters model has 757,000 iters
# 110m parameters model has 762,000 iters
# 350m parameters model has 272,000 iters

block_size = 1024 

batch_size = 4 #16
gradient_accumulation_steps = 32
#max_iters = 5600 # 172394 training samples

# jam_cgpt 350m parameters with block size 256 and 2.15m data size has 308,900 iters
# jam_cgpt 350m parameters with block size 1,024 and 620k data has 129,700 iters
# jam_cgpt 350m parameters with block size 1,024 and 2.15m data has 136,300 iters
# jam_cgpt 350m parameters with block size 1,024 further finetuned with 2.15m data and contextsum data has 137,800 iters
#max_iters = 137500 + 5 * 10 # pretrained with gpt
max_iters = 137900 + 5 * 10 # pretrained with gemini

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
