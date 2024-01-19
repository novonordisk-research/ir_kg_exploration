#!/bin/bash
 
# An example submission script, lines that start with #SBATCH are options for slurm

#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --mem=38G
#SBATCH --mail-type=END
#SBATCH --mail-user=thzo@novonordisk.com
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# and here is you actual job
module purge
module load anaconda3/2021.05
conda activate giannt

which python

GHL_PATH=$1
ALGORITHM=$2
ALGORITHM_KWARGS=$3
PAIRS_FILE=$4
GRAPH_FILE=$5
OUTPUT_FILE=$6
N_CORES=$7
echo $ALGORITHM
echo $ALGORITHM_KWARGS
echo $PAIRS_FILE
echo $GRAPH_FILE
echo $OUTPUT_FILE
echo $N_CORES

$GHL_PATH algorithms calculate_pairwise_feature \
    -g $GRAPH_FILE \
    -p $PAIRS_FILE \
    -o $OUTPUT_FILE \
    -a $ALGORITHM \
    --algorithm_kwargs $ALGORITHM_KWARGS \
    -n $N_CORES
