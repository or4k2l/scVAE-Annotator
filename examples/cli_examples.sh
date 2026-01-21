#!/bin/bash
#
# Example: Command-line usage of scVAE-Annotator
#
# This script demonstrates how to use the scvae-annotate CLI tool.
#

# Example 1: Basic usage with a CSV file
# scvae-annotate --input data/counts.csv --output results/annotations.csv

# Example 2: Using a pre-trained model
# scvae-annotate --input data/counts.csv --model models/pbmc.h5 --output results/annotations.csv

# Example 3: Customizing training parameters
# scvae-annotate --input data/counts.csv --output results/annotations.csv --epochs 200 --latent-dim 15 --batch-size 256

# Example 4: With transposed data (genes as rows)
# scvae-annotate --input data/counts.tsv --output results/annotations.csv --transpose

# For more options, run:
# scvae-annotate --help
