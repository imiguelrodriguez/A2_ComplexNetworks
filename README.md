# Community Detection in Synthetic Networks

This project analyzes the evolution of community structures in synthetic networks as the **internal connection probability (`prr`)** varies. It applies three different **community detection algorithms**:

- **Louvain Method** (Modularity Maximization)
- **Girvan-Newman Algorithm** (Edge Betweenness)
- **Infomap Algorithm** (Flow-based Clustering)

The analysis includes:
✅ Evolution of the **number of communities** and **modularity**  
✅ **Comparison with ground-truth partitions** using:  
   - **Jaccard Index**
   - **Normalized Mutual Information (NMI)**
   - **Normalized Variation of Information (NVI)**  
✅ **Visualization** of communities at key values of `prr` (`0.02`, `0.16`, `1.00`)  

---

## 🚀 Installation

Ensure you have **Python 3.8+** installed. Run the following command to install all dependencies:

```bash
pip install networkx community scikit-learn scipy matplotlib infomap
