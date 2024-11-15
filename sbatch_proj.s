#!/bin/bash
#SBATCH --job-name=llama_batch_processing  # Job name
#SBATCH --nodes=2                          # Number of nodes
#SBATCH --ntasks-per-node=8                # Tasks per node (adjust as needed)
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task
#SBATCH --time=00:10:00                    # Time limit (4 hours as an example)
#SBATCH --mem=16G                          # Memory per node (adjust as needed)
#SBATCH --output=/scratch/ah5192/comp_ling/slurm_output/out_%A_%a.out  # The output will be saved here. %A will be replaced by the slurm job ID, and %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH --mail-user=ah5192@nyu.edu   # Email address
#SBATCH --mail-type=END               # Send an email when all the instances of this job are completed

module purge                          # unload all currently loaded modules in the environment


/scratch/ah5192/comp_ling/run_proj.bash python /scratch/ah5192/comp_ling/comp_ling_proj/src/scripts/run_llama.py --data_dir "/scratch/ah5192/comp_ling/comp_ling_proj/data/tscc_split/tiny" --output_dir "/scratch/ah5192/comp_ling/slurm_output"


