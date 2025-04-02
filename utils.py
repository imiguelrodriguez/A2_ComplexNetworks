import networkx as nx
import community as community_louvain  
from networkx.algorithms.community import girvan_newman
from infomap import Infomap  
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, normalized_mutual_info_score
from scipy.stats import entropy

def analyze_network_evolution(method="louvain"):
    results = []

    for i in range(0, 101, 2):
        prr = i / 100
        file_path = f"A3_synthetic_networks/synthetic_network_N_300_blocks_5_prr_{prr:.2f}_prs_0.02.net"
        G = _load_network(file_path)

        print(f"Processing network: {file_path}")

        # Get the ground-truth partition
        true_partition = _get_true_partition(G)

        # Run the selected community detection method
        if method == "louvain":
            partition, num_communities = _louvain(G)
        elif method == "girvan_newman":
            partition, num_communities = _girvan_newman(G)
        elif method == "infomap":
            partition, num_communities = _infomap(G)
        else:
            raise ValueError("Method must be 'louvain', 'girvan_newman', or 'infomap'.")

        # Compute modularity
        modularity = community_louvain.modularity(partition, G)

        # Compute similarity metrics
        jaccard, nmi, nvi = _compare_partitions(true_partition, partition, G)

        # Store results
        results.append({
            "prr": prr,
            "num_communities": num_communities,
            "modularity": modularity,
            "jaccard": jaccard,
            "nmi": nmi,
            "nvi": nvi,
        })

        print(f"prr={prr:.2f} | Communities: {num_communities} | Modularity: {modularity:.4f} | "
              f"Jaccard: {jaccard:.4f} | NMI: {nmi:.4f} | NVI: {nvi:.4f}")

    return results


def _load_network(file_path=""):
    G = nx.read_pajek(file_path)

    if nx.is_directed(G):
        G = G.to_undirected()

    if G.is_multigraph():
        G = nx.Graph(G)

    return nx.relabel_nodes(G, {node: int(node) for node in G.nodes()})


def _get_true_partition(G):
    """Returns the ground-truth partition (5 blocks of 60 nodes)."""
    true_partition = {}
    for node in G.nodes():
        true_partition[node] = (node - 1) // 60
    return true_partition


def _louvain(G):
    partition = community_louvain.best_partition(G)
    num_communities = len(set(partition.values()))
    return partition, num_communities


def _girvan_newman(G):
    communities = list(girvan_newman(G))
    if communities:
        first_partition = tuple(sorted(c) for c in next(iter(communities)))
        num_communities = len(first_partition)
        partition = {node: i for i, comm in enumerate(first_partition) for node in comm}
        return partition, num_communities
    return {}, 0


def _infomap(G):
    im = Infomap()
    for node in G.nodes():
        im.add_node(node)
    for u, v in G.edges():
        im.add_link(u, v)
    im.run()
    partition = im.get_modules()
    num_communities = len(set(partition.values()))
    return partition, num_communities


def _compare_partitions(true_partition, detected_partition, G):
    """Computes Jaccard Index, NMI, and Normalized Variation of Information (NVI)."""
    true_labels = [true_partition[node] for node in sorted(G.nodes())]
    detected_labels = [detected_partition.get(node, -1) for node in sorted(G.nodes())]

    jaccard = jaccard_score(true_labels, detected_labels, average="macro")
    nmi = normalized_mutual_info_score(true_labels, detected_labels)

    # Compute normalized variation of information (NVI)
    true_clusters = set(true_partition.values())
    detected_clusters = set(detected_partition.values())

    # Create aligned probability distributions
    all_clusters = true_clusters.union(detected_clusters)
    p1 = [list(true_partition.values()).count(c) / len(true_partition) for c in all_clusters]
    p2 = [list(detected_partition.values()).count(c) / len(detected_partition) for c in all_clusters]

    nvi = entropy(p1, p2)

    return jaccard, nmi, nvi

def plot_results(results, method):
    """Plots the evolution of metrics over prr with improved visualization and numerical labels."""
    prr_values = [r["prr"] for r in results]
    num_communities = [r["num_communities"] for r in results]
    modularity = [r["modularity"] for r in results]
    jaccard = [r["jaccard"] for r in results]
    nmi = [r["nmi"] for r in results]
    nvi = [r["nvi"] for r in results]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Primary y-axis (Number of Communities + Modularity)
    ax1.set_xlabel("prr", fontsize=12)
    ax1.set_ylabel("Number of Communities / Modularity", fontsize=12)
    line1, = ax1.plot(prr_values, num_communities, label="Number of Communities", linestyle="--", marker="o", alpha=0.7)
    line2, = ax1.plot(prr_values, modularity, label="Modularity", linestyle="-", marker="s", alpha=0.7)
    ax1.tick_params(axis="y")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Add numerical values along the lines
    for i in range(0, len(prr_values), 10):  
        ax1.text(prr_values[i], num_communities[i], f"{num_communities[i]}", fontsize=9, ha="right", color="tab:blue")
        ax1.text(prr_values[i], modularity[i], f"{modularity[i]:.2f}", fontsize=9, ha="left", color="tab:cyan")

    # Secondary y-axis (Similarity Metrics)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Similarity Metrics", fontsize=12)
    line3, = ax2.plot(prr_values, jaccard, label="Jaccard", color="tab:red", linestyle="-.", marker="^", alpha=0.7)
    line4, = ax2.plot(prr_values, nmi, label="NMI", color="tab:orange", linestyle="-", marker="v", alpha=0.7)
    line5, = ax2.plot(prr_values, nvi, label="NVI", color="tab:pink", linestyle=":", marker="D", alpha=0.7)
    ax2.tick_params(axis="y")

    # Add numerical values along the lines (only for some points)
    for i in range(0, len(prr_values), 5):
        ax2.text(prr_values[i], jaccard[i], f"{jaccard[i]:.2f}", fontsize=9, ha="right", color="tab:red")
        ax2.text(prr_values[i], nmi[i], f"{nmi[i]:.2f}", fontsize=9, ha="left", color="tab:orange")
        ax2.text(prr_values[i], nvi[i], f"{nvi[i]:.2f}", fontsize=9, ha="left", color="tab:pink")

    fig.suptitle(f"Community Evolution - {method.capitalize()} Method", fontsize=14, fontweight="bold")
    ax1.legend(handles=[line1, line2], loc="upper left", fontsize=10, frameon=True)
    ax2.legend(handles=[line3, line4, line5], loc="upper right", fontsize=10, frameon=True)

    plt.show()

def visualize_communities(method="louvain"):
    prr_values = [0.02, 0.16, 1.00]

    # Compute node positions from prr=1.00 network
    G_layout = _load_network("A3_synthetic_networks/synthetic_network_N_300_blocks_5_prr_1.00_prs_0.02.net")
    pos = nx.spring_layout(G_layout, seed=42)  # Using Fruchterman-Reingold layout

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, prr in enumerate(prr_values):
        file_path = f"A3_synthetic_networks/synthetic_network_N_300_blocks_5_prr_{prr:.2f}_prs_0.02.net"
        G = _load_network(file_path)

        # Detect communities
        if method == "louvain":
            partition, _ = _louvain(G)
        elif method == "girvan_newman":
            partition, _ = _girvan_newman(G)
        elif method == "infomap":
            partition, _ = _infomap(G)
        else:
            raise ValueError("Method must be 'louvain', 'girvan_newman', or 'infomap'.")

        # Assign colors based on communities
        communities = list(set(partition.values()))
        cmap = plt.get_cmap("tab10", len(communities))  # Use a colormap with distinct colors
        node_colors = [cmap(communities.index(partition[node])) for node in G.nodes()]

        # Plot network
        ax = axes[idx]
        ax.set_title(f"prr = {prr:.2f}", fontsize=14, fontweight="bold")
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=60, edge_color="gray", alpha=0.6, with_labels=False)

    fig.suptitle(f"Community Structure Visualization - {method.capitalize()} Method", fontsize=16, fontweight="bold")
    plt.show()
