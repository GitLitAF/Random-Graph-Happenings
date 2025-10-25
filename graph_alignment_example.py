"""
Graph Alignment using Approximate IsoRank

This module demonstrates probabilistic/similarity-based graph alignment
for graphs that represent the same underlying semantic space but may be
from disparate sources with varying degrees of overlap.

Key Concepts:
- Local Alignment: Finding best overlapping regions between graphs
- Node Similarity Matrix: Like substitution matrices in sequence alignment
- Gap Penalties: Cost for unaligned nodes (spacer nodes)
- Quality Metrics: Coverage, alignment score, confidence

Alignment Algorithm (IsoRank):
The IsoRank algorithm computes a similarity matrix R where R[i,j] represents
the probability that node i from graph1 should align to node j from graph2.

R is computed iteratively as:
    R = (1-α) * (P2 @ R @ P1^T) + α * E

Where:
- P1, P2: Normalized adjacency matrices (transition probabilities)
- E: Initial node similarity matrix (your prior knowledge)
- α: Balance between topology (0) and node features (1)
- R: Final alignment similarity matrix

After computing R, greedy assignment finds the best one-to-one mappings.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt


def create_node_similarity_matrix(
    graph1: nx.Graph,
    graph2: nx.Graph,
    similarity_func: Optional[callable] = None,
    default_similarity: float = 0.1
) -> np.ndarray:
    """
    Create initial node similarity matrix E.

    This is analogous to a substitution matrix in sequence alignment.
    You can provide custom similarity based on node features, embeddings,
    or semantic similarity.

    Parameters:
    -----------
    graph1, graph2 : networkx.Graph
        The two graphs to align
    similarity_func : callable, optional
        Function(node1_attrs, node2_attrs) -> float [0, 1]
        Computes similarity between two nodes based on their attributes
    default_similarity : float
        Default similarity when no function provided

    Returns:
    --------
    E : np.ndarray of shape (|G1|, |G2|)
        Node similarity matrix
    """
    nodes1 = list(graph1.nodes())
    nodes2 = list(graph2.nodes())

    E = np.zeros((len(nodes1), len(nodes2)))

    for i, n1 in enumerate(nodes1):
        for j, n2 in enumerate(nodes2):
            if similarity_func:
                # Use custom similarity (e.g., based on embeddings, features)
                attrs1 = graph1.nodes[n1]
                attrs2 = graph2.nodes[n2]
                E[i, j] = similarity_func(attrs1, attrs2)
            else:
                # Default: higher similarity for nodes with same degree
                # (This is just an example - you'd use better features)
                deg_sim = 1.0 / (1.0 + abs(graph1.degree[n1] - graph2.degree[n2]))
                E[i, j] = deg_sim * default_similarity

    # Normalize
    if E.sum() > 0:
        E = E / E.sum()

    return E


def compute_isorank_alignment(
    graph1: nx.Graph,
    graph2: nx.Graph,
    node_similarity: np.ndarray,
    alpha: float = 0.7,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-6
) -> Tuple[np.ndarray, Dict]:
    """
    Compute IsoRank alignment similarity matrix.

    Parameters:
    -----------
    graph1, graph2 : networkx.Graph
        Graphs to align
    node_similarity : np.ndarray
        Initial node similarity matrix E
    alpha : float in [0, 1]
        Weight between topology (0) and node features (1)
        Typical value: 0.7
    max_iterations : int
        Number of iterations for convergence
    convergence_threshold : float
        Stop if change < threshold

    Returns:
    --------
    R : np.ndarray
        Final alignment similarity matrix
    info : dict
        Additional information (iterations, convergence, etc.)
    """
    # Get adjacency matrices
    nodes1 = list(graph1.nodes())
    nodes2 = list(graph2.nodes())

    A1 = nx.to_numpy_array(graph1, nodelist=nodes1)
    A2 = nx.to_numpy_array(graph2, nodelist=nodes2)

    # Normalize to get transition matrices
    d1 = A1.sum(axis=1, keepdims=True)
    d2 = A2.sum(axis=1, keepdims=True)

    # Avoid division by zero for isolated nodes
    d1[d1 == 0] = 1
    d2[d2 == 0] = 1

    P1 = A1 / d1.T
    P2 = A2 / d2.T

    # Initialize R
    E = node_similarity.copy()
    if E.sum() == 0:
        E = np.ones_like(E) / E.size
    else:
        E = E / E.sum()

    # Initial guess based on degree distribution
    d = (d1 @ d2.T)
    d = d / (d1.sum() * d2.sum())

    R = (1 - alpha) * d + alpha * E
    R = R.T  # Transpose for computation
    E = E.T

    # Iterative refinement
    iterations = 0
    for i in range(max_iterations):
        R_old = R.copy()
        R = (1 - alpha) * (P2 @ R @ P1.T) + alpha * E

        # Check convergence
        change = np.linalg.norm(R - R_old)
        iterations = i + 1

        if change < convergence_threshold:
            break

    R = R.T  # Transpose back

    info = {
        'iterations': iterations,
        'converged': iterations < max_iterations,
        'final_change': change if iterations > 0 else 0.0
    }

    return R, info


def greedy_alignment(
    R: np.ndarray,
    num_pairs: Optional[int] = None,
    threshold: Optional[float] = None
) -> List[Tuple[int, int, float]]:
    """
    Compute greedy one-to-one node alignment from similarity matrix.

    Parameters:
    -----------
    R : np.ndarray
        Alignment similarity matrix
    num_pairs : int, optional
        Maximum number of pairs to return
    threshold : float, optional
        Only return pairs with similarity >= threshold

    Returns:
    --------
    alignments : List[(idx1, idx2, score)]
        List of (graph1_idx, graph2_idx, similarity_score)
    """
    aligned = []
    R_copy = R.copy()

    max_pairs = min(num_pairs or R.shape[0], *R.shape)

    while len(aligned) < max_pairs:
        # Find best remaining match
        max_val = R_copy.max()

        if threshold and max_val < threshold:
            break

        max_idx = np.unravel_index(R_copy.argmax(), R_copy.shape)
        i, j = max_idx

        aligned.append((i, j, max_val))

        # Remove this row and column from consideration
        R_copy[i, :] = -1
        R_copy[:, j] = -1

    return aligned


def compute_alignment_quality(
    alignments: List[Tuple[int, int, float]],
    graph1: nx.Graph,
    graph2: nx.Graph
) -> Dict:
    """
    Compute quality metrics for graph alignment.

    Metrics:
    - Coverage: Fraction of nodes aligned
    - Edge Correctness: Fraction of edges preserved
    - Mean Similarity: Average alignment confidence

    Parameters:
    -----------
    alignments : List[(idx1, idx2, score)]
        Alignment from greedy_alignment
    graph1, graph2 : networkx.Graph
        The aligned graphs

    Returns:
    --------
    metrics : dict
        Quality metrics
    """
    nodes1 = list(graph1.nodes())
    nodes2 = list(graph2.nodes())

    # Coverage
    coverage1 = len(alignments) / len(nodes1) if nodes1 else 0
    coverage2 = len(alignments) / len(nodes2) if nodes2 else 0

    # Mean similarity
    mean_similarity = np.mean([score for _, _, score in alignments]) if alignments else 0

    # Edge correctness
    edge_correctness = 0
    total_edges = 0

    for i, j, _ in alignments:
        node1 = nodes1[i]
        node2 = nodes2[j]

        # Check neighbors
        neighbors1 = set(graph1.neighbors(node1))
        neighbors2 = set(graph2.neighbors(node2))

        # Find which neighbors are also aligned
        aligned_dict = {nodes1[x]: nodes2[y] for x, y, _ in alignments}

        for n1 in neighbors1:
            if n1 in aligned_dict:
                n2_aligned = aligned_dict[n1]
                if n2_aligned in neighbors2:
                    edge_correctness += 1
                total_edges += 1

    edge_correctness_ratio = edge_correctness / total_edges if total_edges > 0 else 0

    return {
        'coverage_graph1': coverage1,
        'coverage_graph2': coverage2,
        'mean_similarity': mean_similarity,
        'edge_correctness': edge_correctness_ratio,
        'num_alignments': len(alignments)
    }


# Example usage
if __name__ == "__main__":
    # Create two similar but not identical graphs
    # These represent the same underlying semantic space but from different sources

    # Graph 1: Original network
    G1 = nx.Graph()
    G1.add_edges_from([
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)
    ])

    # Graph 2: Perturbed version (some edges different, some nodes missing/extra)
    G2 = nx.Graph()
    G2.add_edges_from([
        (0, 1), (0, 2), (1, 2), (1, 3), (3, 5), (5, 6)
    ])

    print("Graph 1:", G1.number_of_nodes(), "nodes,", G1.number_of_edges(), "edges")
    print("Graph 2:", G2.number_of_nodes(), "nodes,", G2.number_of_edges(), "edges")

    # Create node similarity matrix
    E = create_node_similarity_matrix(G1, G2)

    print("\nNode Similarity Matrix E:")
    print(E)

    # Compute alignment
    R, info = compute_isorank_alignment(G1, G2, E, alpha=0.7, max_iterations=20)

    print(f"\nAlignment converged in {info['iterations']} iterations")
    print(f"Final change: {info['final_change']:.6e}")

    print("\nAlignment Similarity Matrix R:")
    print(R)

    # Get best alignments
    alignments = greedy_alignment(R, num_pairs=min(len(G1), len(G2)))

    print("\nTop alignments (graph1_node, graph2_node, similarity):")
    for i, j, score in alignments[:10]:
        print(f"  Node {i} -> Node {j}: {score:.4f}")

    # Quality metrics
    quality = compute_alignment_quality(alignments, G1, G2)

    print("\nAlignment Quality Metrics:")
    for metric, value in quality.items():
        print(f"  {metric}: {value:.4f}")
