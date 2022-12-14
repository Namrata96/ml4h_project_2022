#!/bin/bash
#SBATCH --job-name=gpt2_ft_eval_exp_letter_1_processed_modified_letter_opt_trained
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --mem=64GB 
#SBATCH --time=1:00:00 
#SBATCH --gres=gpu:1

module purge

singularity exec --nv \
	    --overlay /scratch/as14770/my_new_singularity/my_new_singularity.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python evaluate.py"