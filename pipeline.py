#!/bin/bash
from submitter import CCSubmitter

lr = 0.02
max_epoch = 2
envs = ["export PYTHONOPTIMIZE=1", "export OMP_NUM_THREADS=1"]
save_dir = "runs/0603"
batch_size = 64
num_batches = 10
account = "rrg-mpederso"
mixed_train = True


def create_mixed_train(enable=True):
    if enable:
        return "--enable-scale"
    return ""


baseline_string = f"python train.py --enable_grad_4_extractor --save_dir={save_dir}/baseline --max_epoch={max_epoch} " \
                  f"--lr={lr} --batch_size={batch_size} --num_batches={num_batches} {create_mixed_train(mixed_train)}"
pretrained_moclr_string = f"python pretrain.py --save_dir {save_dir}/pretrained_moclr --max_epoch={max_epoch} --lr={lr} " \
                          f"--batch_size={batch_size} --num_batches={num_batches} {create_mixed_train(mixed_train)}"
pretrained_simclr_stirng = f"python pretrain.py --save_dir {save_dir}/pretrained_simclr --max_epoch={max_epoch} --lr={lr}" \
                           f"  --batch_size={batch_size} --num_batches={num_batches} {create_mixed_train(mixed_train)}"

finetune_moclr_string = f"python train.py --pretrained_checkpoint={save_dir}/pretrained_moclr/pretrain.pth " \
                        f"--save_dir={save_dir}/finetune_moclr --max_epoch={max_epoch}   --lr={lr}" \
                        f" --batch_size={batch_size} --num_batches={num_batches}  {create_mixed_train(mixed_train)}"
finetune_simclr_string = f"python train.py --pretrained_checkpoint={save_dir}/pretrained_simclr/pretrain.pth " \
                         f"--save_dir={save_dir}/finetune_simclr --max_epoch={max_epoch}   --lr={lr} " \
                         f"--batch_size={batch_size} --num_batches={num_batches}  {create_mixed_train(mixed_train)}"

submitter = CCSubmitter(work_dir="./", stop_on_error=True)
submitter.configure_environment(envs)
submitter.configure_sbatch(mem=24, time=4, gres="gpu:4", cpus_per_task=24, account=account)
for job in [baseline_string, pretrained_moclr_string, pretrained_simclr_stirng,
            finetune_moclr_string, finetune_moclr_string]:
    submitter.submit(job, on_local=True, verbose=True)
