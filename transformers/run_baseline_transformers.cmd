#!/bin/bash
#SBATCH --job-name=BASE_TRANSFORMERS
#SBATCH --mail-user=bhtang2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_24h
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=2
#SBATCH --output=/research/d7/fyp25/bhtang2/mad_graph/CS_FYP_sem2/slurm_output/baseline/soc-Qwen2.5-7B_%j.txt 
#SBATCH --gres=gpu:1

MODEL="Qwen/Qwen2.5-7B-Instruct"
CSV_PATH="../dataset/sociology.csv"
SEED="42"
DEVICE="cuda"
QUANTIZATION="fp16"

export CUDA_VISIBLE_DEVICES=0,1
export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
export HF_HOME=/research/d7/fyp25/bhtang2/mad_graph/cache/huggingface
export TRANSFORMERS_CACHE=/research/d7/fyp25/bhtang2/mad_graph/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/research/d7/fyp25/bhtang2/mad_graph/cache/huggingface
export TORCH_HOME=/research/d7/fyp25/bhtang2/mad_graph/cache/torch
export LD_LIBRARY_PATH=/research/d7/fyp25/bhtang2/conda_envs/mad/lib:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mad

# Run from the parent directory so output goes to results/baseline gracefully
python baseline_cyclic_eval_transformers.py --model ${MODEL} --input ${CSV_PATH} --seed ${SEED} --device ${DEVICE} --quantization ${QUANTIZATION}
