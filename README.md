# Qlustering: Harnessing Network-Based Quantum Transport for Data Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![MATLAB](https://img.shields.io/badge/MATLAB-R2024b-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/<your-username>/<your-repo>)

## Overview
Qlustering is a quantum clustering algorithm that leverages quantum transport to reveal intrinsic data structures.  
Data are encoded into quantum state vectors and propagated through a pretrained network, where transport dynamics enhance the separation between data groups.  
This coherence-assisted approach avoids explicit gate-based quantum operations, offering an efficient and hardware-feasible alternative to distance- or kernel-based clustering methods.

## Usage
Run the main script: Qlustering_example
Adjust parameters in config.m to:
Select dataset type
Change network architecture
Modify clustering settings

To run the k means algorithm used in the paper, use Kmeans_iris.m
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
