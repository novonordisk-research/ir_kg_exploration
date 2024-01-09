#!/bin/bash

# --- CHANGE THIS 
ANNGEL_PATH="/nfs_home/users/thzo/.conda/envs/giannt/bin/anngel"
# ---


OUTDIR=`pwd`/2_outputs/nodewise
mkdir -p $OUTDIR

algorithms="betweenness_centrality katz_centrality load_centrality"

for algorithm in $algorithms
do
    algorithm_kwargs="{}"
    graph_file=`pwd`/tmp/s3_files/graph_obl_undirected.pkl
    output_file=$OUTDIR/$algorithm.csv#

    sbatch calc_node_feature.sh $ANNGEL_PATH $algorithm $algorithm_kwargs $graph_file $output_file
done