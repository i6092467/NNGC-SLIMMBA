# Granger-causal Inference in Time Series for Identifying Molecular Fingerprints during Sleep

This repository contains the code implemented within the SLIMMBA project (Sleep and Light Induced
Metabolism Monitored by Breath Analysis). We investigated dependencies between metabolism and sleep using the concept of Granger causality and designed a technique for inferring nonlinear Granger causality based on neural networks and bootstrapping.

Generally, the problem that we considered can be formalised as follows. We assume that we are given *N* replicates of multivariate time series retrieved from different experimental units &mdash; in this case, single individuals, i.e. subjects. These multivariate time series include:
- A categorically-valued target variable *Y* which represents the  sleep stage across *T* time steps.
- *M* continuously-valued ion intensity time series *X<sup>j</sup>*, where *j=1,...,M*.

The goal is then to identify metabolites that are associated with sleep stages, i.e. metabolites that *drive* the sleep stage and metabolites that are *driven by* the stage.

## Contributors

- Ričards Marcinkevičs ([ricards.marcinkevics@inf.ethz.ch](mailto:ricards.marcinkevics@inf.ethz.ch))

- Đorđe Miladinović ([djordje.miladinovic@inf.ethz.ch](mailto:djordje.miladinovic@inf.ethz.ch))

## Background

The relationship between human sleep and metabolism has not yet been studied systematically and well understood. We investigated the association between sleep stages and exhaled breath mass spectrometry in the framework of Granger causality. 

We used a scalable neural network approach for inferring nonlinear Granger causality between continuously- and categorically-valued variables. We tested this technique on a wide range of simulated datasets with differing degrees of nonlinearity and demonstrate that, in many settings, it outperforms the conventional linear vector autoregressive model (VAR). The datasets, on which validation was performed, include (but are not limited to) the Lorenz 96 system and rich and realistic simulations of fMRI time series. 

By leveraging the developed method and the bootstrapping technique, we then identified Granger causes and effects of sleep phase transitions from breathomics data.

A schematic depiction of the implemented neural network architecture for inferring nonlinear Granger causality &mdash; the *Granger causal multilayer perceptron* (GC-MLP):

<p align="center">
  <img src="https://github.com/i6092467/NNGC-SLIMMBA/blob/master/images/Causal_MLP.png" width="514">
</p>

Further details, findings, and results can be found in the [poster](https://github.com/i6092467/NNGC-SLIMMBA/blob/master/images/GC_MS.png) and R. Marcinkevičs' [master thesis](https://github.com/i6092467/NNGC-SLIMMBA/blob/master/documents/Master_Thesis_RMarcinkevics.pdf).

### References

Below we provide a few references helpful for understanding the methodology:
- C. W. J. Granger. Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3):424–438, 1969.
- A. Arnold, Y. Liu, and N. Abe. Temporal causal modeling with graphical Granger methods. In *Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, KDD ’07, pages 66–75, 2007.
- A. Montalto, S. Stramaglia, L. Faes, G. Tessitore, R. Prevete, and D. Marinazzo. Neural networks with non-uniform embedding and explicit validation phase to assess Granger causality. *Neural Networks*, 71:159–171, 2015.
- A. Tank, I. Covert, N. Foti, A. Shojaie, and E. Fox. Neural Granger causality for nonlinear time series, 2018. arXiv:1802.05842.

## Getting started

To run the code, the followinng requirements have to be fulfilled.

### Hardware requirements:

- At least 16 GB of RAM
- A GPU supported by CUDA 10.0

### Software requirements:

All necessary libraries are included into the conda environment provided by `environment.yml`. To build it, run:

`conda env create -f environment.yml`

Then activate the environment:

`conda activate NNGC-SLIMMBA`

## Usage

### Utility functions

Several Python files provide utility functions for data handling and pre-processing, synthetic dataset generation, and data visualisation:

- `data_utils.py` contains functions for handling raw data in MATLAB files. 
- `processing_utils.py` contains various functions for data pre-processing:
  - MS data normalisation.
  - Time series standardisation and smoothing.
  - Construction of training datasets for the GC-MLP neural network.
- `model_utils.py` contains functions for generating artificial time series from different autoregressive models with known causal structures.
- `plotting_utils.py` contains functions for plotting ion intensity and sleep stage time series.

### Neural network models

Classes `MLPgc` and `LSTMgc` provide PyTorch implementations of the GC-MLP and GC-LSTM neural network models for the estimation of Granger causality. These classes are accessible in Python files `mlp_gc.py` and `lstm_gc.py`, respectively.   

### MS and sleep data analysis

The main analysis of causal relationships between ion intensity and sleep stage time series can be performed by running `run_mlp_ms.py` script. It implements bootstrapping and the training procedure for the GC-MLP model. The script takes in a single argument &mdash; the name of the output file. Model hyperparameters and other parameters can be adjusted by changing relevant constants in the script. 

Folder `Experiments` contains scripts for running experiments on permuted and synthetic data and cross-validation.

For a detailed description of function arguments, consult the documentation provided in the code. The dataset is publicly available [here](https://www.research-collection.ethz.ch/handle/20.500.11850/422459).
