# Activity 2: Community Detection

This activity aims to understand how community detection algorithms work and some of the recurrent nuances appearing when using these tools to characterize the mesoscale of complex networks. The activity is composed of two parts:


- **Characterization of the community structure of networks with block structure**: In this part, five community detection algorithms are used to analyze the community structure of networks generated according to the _Stochastic Block Model_(SBM). The analysis includes the evolution of the number of communities and the modularity of the partitions found by each algorithm, a color-coded visualization of the community structure, and a conclusion section comparing the performance of each model based on standard indices.

- **Characterization of the community structure of real networks**: In the second part, community detection algorithms are used to analyze the community structure of a real network capturing face-to-face interactions in a primary school in France.


## Installation

Ensure you have **Python 3.8+** installed. Run the following command to install all dependencies:

```bash
pip install networkx community scikit-learn scipy matplotlib infomap
