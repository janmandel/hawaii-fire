#!/bin/bash
# A simple single-core job template for running a Python script
#SBATCH --job-name=ml_sample_job          # Job name
#SBATCH --partition=math-alderaan         # Partition name (adjust if necessary)
#SBATCH --time=96:00:00                   # Maximum runtime (hh:mm:ss)
#SBATCH --ntasks=1                        # Number of tasks (single core)
#SBATCH --cpus-per-task=1                 # Number of CPUs per task (adjust if needed)
#SBATCH --exclusive                       # Ensure this job exclusively uses the node
#SBATCH --output=logs/ml_sample_g_test%j.log         # Log output file (%j will include the job ID)
#SBATCH --error=logs/ml_sample_g_test%j.err          # Error output file (%j will include the job ID)

# Run the Python script
PYTHONUNBUFFERED=1 python ml_sample_generator.py > logs/ml_sample_generator_test.log  

