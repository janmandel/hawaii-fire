#!/bin/bash
# A simple single-core job template for running a Python script
#SBATCH --job-name=ml_sample_job          # Job name
#SBATCH --partition=math-alderaan         # Partition name (adjust if necessary)
#SBATCH --time=96:00:00                    # Maximum runtime (hh:mm:ss)
#SBATCH --ntasks=1                        # Number of tasks (single core)
#SBATCH --output=logs/ml_sample_%j.log         # Log output file (%j will include the job ID)
#SBATCH --error=logs/ml_sample_%j.err          # Error output file (%j will include the job ID)

# Activate the conda environment
eval "$(conda shell.bash hook)"
conda activate hi-fire

PYTHONUNBUFFERED=1 python ml_sample_generator.py > logs/ml_sample.log 2>&1  

