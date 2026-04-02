#!/bin/bash
#SBATCH --job-name=BASE_TRANSFORMERS
#SBATCH --mail-user=bhtang2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_24h
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=2
#SBATCH --output=/research/d7/fyp25/bhtang2/mad_graph/CS_FYP_sem2/slurm_output/sampling/dse-Phi3_%j.txt 
#SBATCH --gres=gpu:1

MODEL="microsoft/Phi-3-mini-4k-instruct"
CSV_PATH="dataset/2012-2020_ICT_DSE.csv"
SEED="42"
DEVICE="cuda"
QUANTIZATION="fp16"

export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
export HF_HOME=/research/d7/fyp25/bhtang2/mad_graph/cache/huggingface
export TRANSFORMERS_CACHE=/research/d7/fyp25/bhtang2/mad_graph/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/research/d7/fyp25/bhtang2/mad_graph/cache/huggingface
export TORCH_HOME=/research/d7/fyp25/bhtang2/mad_graph/cache/torch
export LD_LIBRARY_PATH=/research/d7/fyp25/bhtang2/conda_envs/mad/lib:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mad

python sampling_eval.py --backend transformers --model ${MODEL} \
       --input ${CSV_PATH} --sampling-n 15 \
       --sampling-temp 0.7 --num-workers 4 --quantization ${QUANTIZATION}
