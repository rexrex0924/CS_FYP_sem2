#!/bin/bash
#SBATCH --job-name=MAD_TEST
#SBATCH --mail-user=bhtang2@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu_24h
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=2
#SBATCH --output=/research/d7/fyp25/bhtang2/mad_graph/CS_FYP_sem2/slurm_output/output%j.txt ##Do not use "~" point to your home!
#SBATCH --gres=gpu:1

MODEL="microsoft/Phi-4-reasoning"
CSV_PATH="../dataset/2012-2020_ICT_DSE.csv"
TEMPERATURE="0.7"
SEED="42"
QNUM="100"
DEVICE="cuda"

export SLURM_CONF=/opt1/slurm/gpu-slurm.conf
export HF_HOME=/research/d7/fyp25/bhtang2/mad_graph/cache/huggingface
export TRANSFORMERS_CACHE=/research/d7/fyp25/bhtang2/mad_graph/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/research/d7/fyp25/bhtang2/mad_graph/cache/huggingface
export TORCH_HOME=/research/d7/fyp25/bhtang2/mad_graph/cache/torch

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mad

python mad_graph_analysis_transformers.py --model ${MODEL} --input ${CSV_PATH} --temperature ${TEMPERATURE} --seed ${SEED} --max-questions ${QNUM} --device ${DEVICE}
