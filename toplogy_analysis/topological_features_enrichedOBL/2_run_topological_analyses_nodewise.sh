#!/bin/bash

# --- CHANGE THIS 
ANNGEL_PATH="/nfs_home/users/thzo/.conda/envs/giannt/bin/anngel"
# ---


OUTDIR=`pwd`/2_outputs/nodewise
mkdir -p $OUTDIR

algorithms="degree_centrality in_degree_centrality out_degree_centrality pagerank closeness_centrality betweenness_centrality katz_centrality load_centrality triangles average_neighbor_degree clustering"

for algorithm in $algorithms
do
    algorithm_kwargs="{}"
    graph_file=`pwd`/tmp/s3_files/graph_obl_enriched_undirected.pkl
    output_file=$OUTDIR/$algorithm.csv

    sbatch calc_node_feature.sh $ANNGEL_PATH $algorithm $algorithm_kwargs $graph_file $output_file
done

algorithm="eigenvector_centrality"
algorithm_kwargs="{\"max_iter\":1000,\"tol\":0.0001}"
graph_file=`pwd`/tmp/s3_files/graph_obl_enriched_undirected.pkl
output_file=$OUTDIR/$algorithm.csv
sbatch calc_node_feature.sh $ANNGEL_PATH $algorithm ${algorithm_kwargs} $graph_file $output_file


pairs_file=`pwd`/tmp/s3_files/pairs_obl.pkl
