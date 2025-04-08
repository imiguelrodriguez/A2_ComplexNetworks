import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from networkx.algorithms.community.quality import modularity as modularity_function
from infomap import Infomap
from sklearn.metrics import jaccard_score, normalized_mutual_info_score, confusion_matrix
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment

from enum import Enum, auto

class CommunityDetectionMethod(Enum):
    LOUVAIN = auto()
    GIRVAN_NEWMAN = auto()
    GREEDY = auto()
    INFOMAP = auto()
    LABEL_PROPAGATION = auto()

    def __str__(self):
        return self.name

DIR = "A3_synthetic_networks"


def analyze_network_evolution(method=CommunityDetectionMethod.LOUVAIN):
    results = []

    for i in range(0, 101, 2):
        prr = i / 100
        file_path = f"{DIR}/synthetic_network_N_300_blocks_5_prr_{prr:.2f}_prs_0.02.net"
        G = _load_network(file_path)

        print(f"Processing network: {file_path}")
        true_partition = _get_true_partition(G)

        match method:
            case CommunityDetectionMethod.LOUVAIN:
                partition, num_communities = _louvain(G)
            case CommunityDetectionMethod.GIRVAN_NEWMAN:
                partition, num_communities = _girvan_newman(G)
            case CommunityDetectionMethod.GREEDY:
                partition, num_communities = _greedy(G)
            case CommunityDetectionMethod.INFOMAP:
                partition, num_communities = _infomap(G)
            case CommunityDetectionMethod.LABEL_PROPAGATION:
                partition, num_communities = _label_propagation(G)
            case _:
                raise ValueError(f"Method must be a valid {CommunityDetectionMethod} enumerate class.")

        modularity = modularity_function(G, partition)
        jaccard, nmi, nvi = _compare_partitions(true_partition, partition, G)

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
    true_partition = {}
    for node in G.nodes():
        true_partition[node] = (node - 1) // 60
    return true_partition


def _louvain(G):
    communities = nx.algorithms.community.louvain.louvain_communities(G)
    num_communities = len(communities)
    return communities, num_communities


def _girvan_newman(G):
    communities = nx.algorithms.community.girvan_newman(G)
    if communities:
        partition = tuple(sorted(c) for c in next(communities))
        num_communities = len(partition)
        return partition, num_communities
    return {}, 0


def _greedy(G):
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    return communities, len(communities)


def _infomap(G):
    im = Infomap(silent=True)
    for node in G.nodes():
        im.add_node(node)
    for u, v in G.edges():
        im.add_link(u, v, weight=G[u][v].get("weight", 1.0))
    im.run()
    partition = im.get_modules()
    num_communities = len(set(partition.values()))
    # Convert dict to list of sets
    communities = {}
    for node, module in partition.items():
        communities.setdefault(module, set()).add(node)
    return list(communities.values()), num_communities


def _label_propagation(G):
    communities = nx.algorithms.community.label_propagation_communities(G)
    communities = list(communities)
    return communities, len(communities)


def _align_labels(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_pred = [mapping.get(label, -1) for label in pred_labels]
    return aligned_pred


def _compare_partitions(true_partition, detected_partition, G):
    """Computes Jaccard Index, NMI, and Normalized Variation of Information (NVI)."""
    true_labels = [true_partition[node] for node in sorted(G.nodes())]

    # Convert detected_partition (list of sets) to a dict mapping node -> community label
    detected_partition_dict = {}
    for i, community in enumerate(detected_partition):
        for node in community:
            detected_partition_dict[node] = i

    detected_labels = [detected_partition_dict.get(node, -1) for node in sorted(G.nodes())]

    # Align detected labels to true labels
    aligned_labels = _align_labels(true_labels, detected_labels)

    jaccard = jaccard_score(true_labels, aligned_labels, average="macro")
    nmi = normalized_mutual_info_score(true_labels, aligned_labels)

    # Compute normalized variation of information (NVI)
    true_clusters = set(true_partition.values())
    detected_clusters = set(detected_partition_dict.values())
    all_clusters = true_clusters.union(detected_clusters)
    p1 = [list(true_partition.values()).count(c) / len(true_partition) for c in all_clusters]
    p2 = [list(detected_partition_dict.values()).count(c) / len(detected_partition_dict) for c in all_clusters]
    nvi = entropy(p1, p2)

    return jaccard, nmi, nvi

def plot_results(results, method):
    prr_values = [r["prr"] for r in results]
    num_communities = [r["num_communities"] for r in results]
    modularity = [r["modularity"] for r in results]
    jaccard = [r["jaccard"] for r in results]
    nmi = [r["nmi"] for r in results]
    nvi = [r["nvi"] for r in results]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_xlabel("prr", fontsize=12)
    ax1.set_ylabel("Number of Communities", fontsize=12)
    line1, = ax1.plot(prr_values, num_communities, label="Number of Communities", linestyle="-", marker="o", alpha=0.7)
    ax1.tick_params(axis="y")
    ax1.grid(True, linestyle="--", alpha=0.5)

    for i in range(0, len(prr_values), 10):  
        ax1.text(prr_values[i], num_communities[i], f"{num_communities[i]}", fontsize=9, ha="right", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Similarity Metrics", fontsize=12)
    line2, = ax2.plot(prr_values, modularity, label="Modularity", color="tab:cyan", linestyle="-", marker="s", alpha=0.7)
    line3, = ax2.plot(prr_values, jaccard, label="Jaccard", color="tab:red", linestyle="-.", marker="^", alpha=0.7)
    line4, = ax2.plot(prr_values, nmi, label="NMI", color="tab:orange", linestyle="-", marker="v", alpha=0.7)
    line5, = ax2.plot(prr_values, nvi, label="NVI", color="tab:pink", linestyle=":", marker="D", alpha=0.7)
    ax2.tick_params(axis="y")

    for i in range(0, len(prr_values), 5):
        ax2.text(prr_values[i], modularity[i], f"{modularity[i]:.2f}", fontsize=9, ha="left", color="tab:cyan")
        ax2.text(prr_values[i], jaccard[i], f"{jaccard[i]:.2f}", fontsize=9, ha="right", color="tab:red")
        ax2.text(prr_values[i], nmi[i], f"{nmi[i]:.2f}", fontsize=9, ha="left", color="tab:orange")
        ax2.text(prr_values[i], nvi[i], f"{nvi[i]:.2f}", fontsize=9, ha="left", color="tab:pink")

    fig.suptitle(f"Community Evolution - {method} Method", fontsize=14, fontweight="bold")
    ax1.legend(handles=[line1], loc="upper left", fontsize=10, frameon=True)
    ax2.legend(handles=[line2, line3, line4, line5], loc="upper right", fontsize=10, frameon=True)

    plt.show()


def visualize_communities(method=CommunityDetectionMethod.LOUVAIN):
    prr_values = [0.02, 0.16, 1.00]

    # Compute node positions from prr=1.00 network
    G_layout = _load_network(f"{DIR}/synthetic_network_N_300_blocks_5_prr_1.00_prs_0.02.net")
    pos = nx.spring_layout(G_layout, seed=42)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, prr in enumerate(prr_values):
        file_path = f"{DIR}/synthetic_network_N_300_blocks_5_prr_{prr:.2f}_prs_0.02.net"
        G = _load_network(file_path)
        
        match method:
            case CommunityDetectionMethod.LOUVAIN:
                partition, _ = _louvain(G)
            case CommunityDetectionMethod.GIRVAN_NEWMAN:
                partition, _ = _girvan_newman(G)
            case CommunityDetectionMethod.INFOMAP:
                partition, _ = _infomap(G)
            case CommunityDetectionMethod.GREEDY:
                partition, _ = _greedy(G)
            case CommunityDetectionMethod.LABEL_PROPAGATION:
                partition, _ = _label_propagation(G)
            case _:
                raise ValueError(f"Method must be a valid {CommunityDetectionMethod} enumerate class.")

        node_to_community = {}
        for id, community in enumerate(partition):
            for node in community:
                node_to_community[node] = id

        num_communities = len(partition)
        cmap = plt.get_cmap("tab10", num_communities)
        node_colors = [cmap(node_to_community.get(node, -1)) for node in G.nodes()]
        ax = axes[idx]
        ax.set_title(f"prr = {prr:.2f}", fontsize=14, fontweight="bold")
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=60, edge_color="gray", alpha=0.6, with_labels=False)

    fig.suptitle(f"Community Structure Visualization - {method} Method", fontsize=16, fontweight="bold")
    plt.show()





def visualize_communities_kamada_kawai(network_path: str, method: CommunityDetectionMethod):
    import os

    # Load the target network (weighted or unweighted)
    G = _load_network(network_path)
    weighted = _is_weighted(network_path)

    # Always load the weighted graph to compute the layout
    weighted_path = network_path.replace("primaryschool_u", "primaryschool_w")
    G_weighted = _load_network(weighted_path)

    # Compute inverse weights for Kamada-Kawai
    inv_weights = {(u, v): 1 / d['weight'] if d.get('weight', 1) > 0 else 1.0 for u, v, d in G_weighted.edges(data=True)}
    nx.set_edge_attributes(G_weighted, inv_weights, 'inv_weight')
    pos = nx.kamada_kawai_layout(G_weighted, weight='inv_weight')

    # Detect communities
    match method:
        case CommunityDetectionMethod.LOUVAIN:
            communities = nx.algorithms.community.louvain_communities(G, weight="weight" if weighted else None)
        case CommunityDetectionMethod.GREEDY:
            communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight" if weighted else None)
        case CommunityDetectionMethod.INFOMAP:
            communities, _ = _infomap(G)
        case _:
            raise ValueError(f"Method must be a valid {CommunityDetectionMethod} enumerate class.")

    print(f"Number of communities: {len(communities)}")

    # Create mapping node -> community
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[node] = i

    # Assign colors
    community_ids = sorted(set(partition.values()))
    cmap = plt.get_cmap('tab10', len(community_ids))
    node_colors = [cmap(partition[node]) for node in G.nodes()]

    # Draw the graph
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, alpha=0.4)

    method_name = method.name.capitalize()
    title = f"Community Structure ({method_name}, {'Weighted' if weighted else 'Unweighted'})"
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def stack_plot_school(network_path: str, metadata_path: str, method: CommunityDetectionMethod):
    """
    Plots a stacked bar chart showing the composition of detected communities
    in terms of school groups.

    Parameters:
    - G: networkx Graph
    - communities: list of sets (each set = a community)
    - metadata_path: str, path to metadata file with columns: node, school_group
    - title: str, plot title
    """

    weighted = _is_weighted(network_path)
    G = _load_network(network_path)
    if weighted:
        match method:
            case CommunityDetectionMethod.LOUVAIN:
                communities = nx.algorithms.community.louvain_communities(G, weight="weight")
            case CommunityDetectionMethod.GREEDY:
                communities = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
            case CommunityDetectionMethod.INFOMAP:
                communities, _ = _infomap(G)
            case _:
                raise ValueError(f"Method must be a valid {CommunityDetectionMethod} enumerate class.")
    else:
        match method:
            case CommunityDetectionMethod.LOUVAIN:
                communities = nx.algorithms.community.louvain_communities(G)
            case CommunityDetectionMethod.GREEDY:
                communities = nx.algorithms.community.greedy_modularity_communities(G)
            case CommunityDetectionMethod.INFOMAP:
                communities, _ = _infomap(G)
            case _:
                raise ValueError(f"Method must be a valid {CommunityDetectionMethod} enumerate class.")

    # Load metadata
    metadata = pd.read_csv(metadata_path, sep=r"\s+", header=None, names=["node", "school_group"])
    metadata["node"] = metadata["node"].astype(str)  # Ensure node IDs are strings
    metadata.set_index("node", inplace=True)

    # Build node -> community ID mapping
    partition = {}
    for i, community in enumerate(communities):
        for node in community:
            partition[str(node)] = i  # Ensure string type to match metadata index

    # Build dataframe: node, community, school_group
    data = []
    for node in G.nodes():
        node_str = str(node)
        if node_str in metadata.index:
            group = metadata.loc[node_str, "school_group"]
            community = partition.get(node_str, -1)
            data.append((node_str, group, community))

    df = pd.DataFrame(data, columns=["node", "school_group", "community"])

    # Count: how many from each school group per community
    community_group_counts = df.groupby(["community", "school_group"]).size().unstack(fill_value=0)

    # Plot
    ax = community_group_counts.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="tab20")
    title = "Composition of Detected Communities by School Group - " + method.__str__() + " Weighted" if weighted else "Composition of Detected Communities by School Group - " + method.__str__() + " Unweighted"
    plt.title(title)
    plt.xlabel("Community ID")
    plt.ylabel("Number of Individuals")
    plt.legend(title="School Group", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def _is_weighted(network_path:str)-> bool:
    return False if network_path.split("/")[1].split("_")[1].split(".")[0] == "u" else True
