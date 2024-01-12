# Identification of Insulin Resistance-Related Genes with Biomedical Knowledge Graphs Topology and Embeddings

## Introduction

This repository contains code for identifying insulin resistance-related genes using topological feature engineering, link prediction, positive unlabelled learning (PUL) and outlier detection algorithms. We used OpenBioLink and Hetionet biomedical knowledge graphs (biomedKGs) to predict IR-related genes and evaluated how model performance was affected by the size of the training set and by enriching the biomedKG with IR information. We also assessed the biological relation of the predictions to IR-related processes using the DepMap and Multiscale Interactome biomedKGs datasets and bioinformatic pathway functional annotations.

The link to our paper can be found here: www.???

## Installation

To install this project, please follow these steps:

1. Clone this repository to your local machine
2. Install the required dependencies by running `pip install -r requirements.txt`
3.

## Usage

To use this project, is broken down into the following parts

* Link prediction: Trains the models using link prediction, this can be used then for link prediction task or to extract the embeddings
* Topological Analysis:
* Positive unlabelled:
* Visualisation: Biological characterization and plotting the resuls for our paper
* data: Inlcudes the list of genes identified as IR genes, the relevant disease related to IR and the edited version of OpenBioLink and Hetionet



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
