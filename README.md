# Identification of Insulin Resistance-Related Genes with Biomedical Knowledge Graphs Topology and Embeddings

## Introduction

This repository contains code for identifying insulin resistance-related genes using topological feature engineering, link prediction, positive unlabelled learning (PUL) and outlier detection algorithms. We used OpenBioLink and Hetionet biomedical knowledge graphs (biomedKGs) to predict IR-related genes and evaluated how model performance was affected by the size of the training set and by enriching the biomedKG with IR information. We also assessed the biological relation of the predictions to IR-related processes using the DepMap and Multiscale Interactome biomedKGs datasets and bioinformatic pathway functional annotations.

The paper can be found here: 

## Installation

To install this project, please follow these steps:


1. Clone this repository to your local machine.
2. Install the required dependencies by running pip install -r requirements.txt.
3. Choose one of the tracks: link prediction or topology to generate trained models and embeddings.
4. Use the generated embeddings for the positive unlabelled track.
5. Visualizations can be produced by using the predicted genes and the embeddings for the models1.


## Usage

To use this project, is broken down into the following parts

* [Link prediction](./link_prediction/): Trains the models using link prediction, this can be used then for link prediction task or to extract the embeddings
* [Topological Analysis](./topology_analysis/): Derive the topological feature from the KG using the paper from [https://doi.org/10.1073/pnas.1914950117](https://doi.org/10.1073/pnas.1914950117
* [Positive unlabelled](./positive_unlabelled/): Uses the generated embeddings to perform positive unlabelled learning, a semi-supervised learning technique where only a subset of the data is labeled.
* [Visualisation](./visualisation/): Biological characterization and plotting the resuls for our paper.
* [Data](./data/): Inlcudes the list of genes identified as IR genes, the relevant disease related to IR and the edited version of OpenBioLink and Hetionet.


## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Authors

- Tankred Ott
- Marc Boubnovski Martell
- Viktor Sandberg
- Ramneek Gupta
- Marie Lisandra Zepeda Mendoza

## Acknowledgments

We would like to thank the creators of the OpenBioLink and Hetionet biomedKGs datasets for making their data publicly available. We would also like to thank the contributors to the DepMap and Multiscale Interactome biomedKGs datasets and bioinformatic pathway functional annotations for their work.
