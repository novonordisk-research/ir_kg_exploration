#!/bin/bash
 
# An example submission script, lines that start with #SBATCH are options for slurm

#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=thzo@novonordisk.com
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

# and here is you actual job
module purge
module load anaconda3/2021.05
conda activate giannt

which python

INPUT_DIR=$1
OUTPUT_DIR=$2
N_JOBS=20

python 5_train_ml_reduced.py \
    -i $INPUT_DIR \
    -o $OUTPUT_DIR \
    -n $N_JOBS \
