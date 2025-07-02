#!/bin/bash
#BSUB -J thesis
#BSUB -q gpuv100
#BSUB -W 300
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -o thesis_%J.out
#BSUB -e thesis_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

#Load preinstalled modules
module load python3/3.10.12
module load numpy/1.24.3-python-3.10.12-openblas-0.3.23
module load cudnn/v8.9.1.23-prod-cuda-11.X 
module load pandas/2.0.2-python-3.10.12
module load cuda/11.8

#Activate virtual environment
source /work3/s232855/whisper/bin/activate

ngpu=1  # number of GPUs to perform distributed training on.

export WANDB_API_KEY=$(cat ~/.wandb_key)

torchrun --nproc_per_node=${ngpu} finetune.py \
        --language English \
        --sampling_rate 16000 \
        --num_proc 4 \
        --train_strategy steps \
        --learning_rate 3e-3 \
        --warmup 1000 \
        --train_batchsize 16 \
        --eval_batchsize 8 \
        --output_dir test \
        --num_steps 10000 \
        --resume_from_ckpt None \