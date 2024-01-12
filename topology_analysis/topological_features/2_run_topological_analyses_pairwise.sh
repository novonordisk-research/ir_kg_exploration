#!/bin/bash

# Calculate parwise topological features

# --- CHANGE THIS 
ANNGEL_PATH="PATH TO THE ANNGEL EXECUTABLE"
N_CORES=10
# ---


OUTDIR=`pwd`/2_outputs/pairwise
mkdir -p $OUTDIR

algorithms="n_common_neighbors shortest_path_length adamic_adar_index preferential_attachment jaccard_coefficient resource_allocation_index leicht_holme_newman_index"

for algorithm in $algorithms
do
    algorithm_kwargs="{}"
    graph_file=`pwd`graph_obl_undirected.pkl
    
    suffix="disease"
    pairs_file=`pwd`pairs_obl_$suffix.pkl
    output_file=$OUTDIR/${algorithm}_${suffix}.csv
    sbatch calc_pairwise_feature.sh $ANNGEL_PATH $algorithm $algorithm_kwargs $pairs_file $graph_file $output_file $N_CORES
    
    suffix="pathway"
    pairs_file=`pwd`pairs_obl_$suffix.pkl
    output_file=$OUTDIR/${algorithm}_${suffix}.csv
    sbatch calc_pairwise_feature.sh $ANNGEL_PATH $algorithm $algorithm_kwargs $pairs_file $graph_file $output_file $N_CORES
done

algorithm="personalized_pagerank"
algorithm_kwargs="{}"
graph_file=`pwd`graph_obl_undirected.pkl

suffix="disease"
pairs_file=`pwd`input_pairs_obl_$suffix.pkl
output_file=$OUTDIR/${algorithm}_${suffix}.csv
sbatch calc_pairwise_feature.sh $ANNGEL_PATH $algorithm $algorithm_kwargs $pairs_file $graph_file $output_file $N_CORES

suffix="pathway"
pairs_file=`pwd`input_pairs_obl_$suffix.pkl
output_file=$OUTDIR/${algorithm}_${suffix}.csv
sbatch calc_pairwise_feature.sh $ANNGEL_PATH $algorithm $algorithm_kwargs $pairs_file $graph_file $output_file $N_CORES
