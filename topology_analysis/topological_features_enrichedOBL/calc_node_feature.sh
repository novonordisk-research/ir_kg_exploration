#!/bin/bash
 
# An example submission script, lines that start with #SBATCH are options for slurm

#SBATCH --partition=compute
#SBATCH --time=72:00:00
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=thzo@novonordisk.com
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# and here is you actual job
module purge
module load anaconda3/2021.05
conda activate giannt

which python

ANNGEL_PATH=$1
ALGORITHM=$2
ALGORITHM_KWARGS=$3
INPUT_FILE=$4
OUTPUT_FILE=$5
echo $ALGORITHM
echo $ALGORITHM_KWARGS
echo $INPUT_FILE
echo $OUTPUT_FILE

$ANNGEL_PATH algorithms calculate_node_feature \
    -i $INPUT_FILE \
    -o $OUTPUT_FILE \
    -a $ALGORITHM \
    --algorithm_kwargs $ALGORITHM_KWARGS
