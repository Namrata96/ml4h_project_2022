#!/bin/bash
#SBATCH --job-name=gpt2_finetune1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=8
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --mem=64GB 
#SBATCH --time=24:00:00 
#SBATCH --gres=gpu:1

module purge

singularity exec --nv \
	    --overlay /scratch/as14770/my_new_singularity/my_new_singularity.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	    /bin/bash -c "source /ext3/env.sh; python finetune.py"