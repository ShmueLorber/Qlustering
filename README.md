# Qlustering: Harnessing Network-Based Quantum Transport for Data Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![MATLAB](https://img.shields.io/badge/MATLAB-R2024b-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/<your-username>/<your-repo>)

## Overview
Qlustering is a quantum clustering algorithm that leverages quantum transport to reveal intrinsic data structures.  
Data are encoded into quantum state vectors and propagated through a pretrained network, where transport dynamics enhance the separation between data groups.  
This coherence-assisted approach avoids explicit gate-based quantum operations, offering an efficient and hardware-feasible alternative to distance- or kernel-based clustering methods.

## Usage
Run the main script:
Qlustering_example

Adjust network parameters directly in the code.

Select the dataset type using the following scripts:

- OverlapWaveFunctionGenerator.m – generates the overlap-controlled wavefunction dataset (the first dataset presented in the paper).

- IPRgenerator.m – creates the localization dataset based on the Inverse Participation Ratio (IPR).

- QM9_featureExtraction.py – extracts molecules from the QM9 dataset and computes their SID (Sorted Interatomic Distances), used as input to the Qlustering algorithm. Use this code to save a matlab file of the data set and then extract it to Qlustering_example

- iris.m – loads the Iris dataset. Modify the code to include or exclude specific features: [sepal length, sepal width, petal length, petal width].

To reproduce the classical baseline used in the paper, run:
Kmeans_example.m
## Requirements
MATLAB R2024b or later
Statistics and Machine Learning Toolbox
(Optional) Parallel Computing Toolbox

## If you use this code, please cite:
Shmuel Lorber, Yonatan Dubi. Qlustering: Harnessing Network-Based Quantum Transport for Data Clustering. In preparation.

## License
This project is licensed under the MIT License
## Acknowledgments
We thank Ari Packan at Ben-Gurion University for discussions and feedback during development.
