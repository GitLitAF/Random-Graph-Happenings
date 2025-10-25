"""
Advanced Graph Alignment using the netalign (IsoRank) Package

This demonstrates using the approximate_ISORANK package for aligning
graphs from disparate sources representing the same semantic manifold.

Use cases:
- Aligning knowledge graphs from different sources
- Matching biological networks across species
- Finding correspondences in multi-modal graph representations
- Local alignment with partial overlaps

Key features:
- Fast approximations (R0, R1) for scalability
- Adjustable alpha parameter for topology vs. features
- Quality metrics (edge correctness, coverage, etc.)
"""

import sys
sys.path.insert(0, '/home/user/Random-Graph-Happenings/approximate_ISORANK')

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional


def prepare_graphs_for_isorank(
    graph1: nx.Graph,
    graph2: nx.Graph,
    output_dir: str = "temp_alignment"
) -> Tuple[str, str]:
    """
    Convert NetworkX graphs to edge list format for IsoRank.

    Parameters:
    -----------
    graph1, graph2 : networkx.Graph
    output_dir : str

    Returns:
    --------
    (path_to_net1, path_to_net2)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Write edge lists (tab-delimited, no header)
    net1_path = f"{output_dir}/net1.tsv"
    net2_path = f"{output_dir}/net2.tsv"

    # Convert to edge dataframe
    edges1 = [[u, v] for u, v in graph1.edges()]
    edges2 = [[u, v] for u, v in graph2.edges()]

    df1 = pd.DataFrame(edges1)
    df2 = pd.DataFrame(edges2)

    df1.to_csv(net1_path, sep='\t', header=False, index=False)
    df2.to_csv(net2_path, sep='\t', header=False, index=False)

    return net1_path, net2_path


def create_similarity_scores(
    graph1: nx.Graph,
    graph2: nx.Graph,
    similarity_func: Optional[callable] = None,
    output_path: str = "temp_alignment/similarity.tsv"
) -> str:
    """
    Create node similarity scores file for IsoRank.

    This is like the BLAST scores in bioinformatics - representing
    prior knowledge about which nodes might correspond.

    File format (tab-delimited):
    node1_id  node2_id  similarity_score

    Parameters:
    -----------
    graph1, graph2 : networkx.Graph
    similarity_func : callable, optional
        Function(node1_id, node2_id, G1, G2) -> float
    output_path : str

    Returns:
    --------
    path to similarity file
    """
    nodes1 = list(graph1.nodes())
    nodes2 = list(graph2.nodes())

    similarities = []

    for n1 in nodes1:
        for n2 in nodes2:
            if similarity_func:
                score = similarity_func(n1, n2, graph1, graph2)
            else:
                # Default: structural similarity based on degree
                deg1 = graph1.degree[n1]
                deg2 = graph2.degree[n2]
                score = 1.0 / (1.0 + abs(deg1 - deg2))

            if score > 0.01:  # Only save non-trivial similarities
                similarities.append({
                    'net1': n1,
                    'net2': n2,
                    'score': score
                })

    df = pd.DataFrame(similarities)
    df.to_csv(output_path, sep='\t', index=False)

    return output_path


def run_isorank_alignment(
    net1_path: str,
    net2_path: str,
    similarity_path: str,
    alpha: float = 0.7,
    niter: int = 1,
    npairs: int = 1000,
    output_path: str = "temp_alignment/alignment.tsv"
) -> pd.DataFrame:
    """
    Run IsoRank alignment using the netalign package.

    Parameters:
    -----------
    net1_path, net2_path : str
        Paths to edge list files
    similarity_path : str
        Path to node similarity scores
    alpha : float in [0, 1]
        Balance parameter:
        - 0.0: Pure topology (ignore node features)
        - 1.0: Pure node similarity (ignore topology)
        - 0.6-0.7: Recommended balance
    niter : int
        Approximation level:
        - 0: R0 approximation (fastest, least accurate)
        - 1: R1 approximation (fast, good accuracy)
        - 10+: Full IsoRank (slower, best accuracy)
    npairs : int
        Number of aligned pairs to return
    output_path : str
        Where to save alignment results

    Returns:
    --------
    alignment_df : pd.DataFrame
        Columns: [net1_node, net2_node]
    """
    from netalign.approx_isorank.io_utils import compute_adjacency, compute_pairs
    from netalign.approx_isorank.isorank_compute import compute_isorank, compute_greedy_assignment

    # Load networks
    df1 = pd.read_csv(net1_path, sep="\t", header=None)
    df2 = pd.read_csv(net2_path, sep="\t", header=None)
    dpairs = pd.read_csv(similarity_path, sep="\t")

    # Extract network names from paths
    org1 = "net1"
    org2 = "net2"

    print(f"Computing adjacency matrices...")
    Af1, nA1 = compute_adjacency(df1)
    Af2, nA2 = compute_adjacency(df2)

    print(f"Graph 1: {Af1.shape[0]} nodes")
    print(f"Graph 2: {Af2.shape[0]} nodes")

    print(f"Creating similarity matrix...")
    E = compute_pairs(dpairs, nA1, nA2, org1, org2)

    print(f"Computing IsoRank similarity (alpha={alpha}, iterations={niter})...")
    R = compute_isorank(Af1, Af2, E, alpha=alpha, maxiter=niter)[-1]

    print(f"Performing greedy alignment...")
    pairs = compute_greedy_assignment(R, npairs)

    # Map indices back to original node IDs
    rnA1 = {v: k for k, v in nA1.items()}
    rnA2 = {v: k for k, v in nA2.items()}

    results = pd.DataFrame(pairs, columns=['idx1', 'idx2'])
    results['node1'] = results['idx1'].apply(lambda x: rnA1[x])
    results['node2'] = results['idx2'].apply(lambda x: rnA2[x])

    # Get alignment scores from R matrix
    results['score'] = results.apply(lambda row: R[row['idx1'], row['idx2']], axis=1)

    results = results[['node1', 'node2', 'score']]
    results.to_csv(output_path, sep='\t', index=False)

    print(f"Alignment saved to {output_path}")

    return results


def evaluate_alignment(
    alignment_df: pd.DataFrame,
    graph1: nx.Graph,
    graph2: nx.Graph
) -> Dict:
    """
    Evaluate alignment quality.

    Metrics:
    - Coverage: What fraction of nodes were aligned?
    - Edge correctness: How many edges are preserved?
    - Mean confidence: Average alignment score
    """
    n_aligned = len(alignment_df)
    coverage1 = n_aligned / graph1.number_of_nodes() if graph1.number_of_nodes() > 0 else 0
    coverage2 = n_aligned / graph2.number_of_nodes() if graph2.number_of_nodes() > 0 else 0

    mean_score = alignment_df['score'].mean()

    # Edge correctness
    aligned_map = dict(zip(alignment_df['node1'], alignment_df['node2']))
    correct_edges = 0
    total_testable = 0

    for n1, n2 in aligned_map.items():
        if n1 not in graph1 or n2 not in graph2:
            continue

        neighbors1 = set(graph1.neighbors(n1))
        neighbors2 = set(graph2.neighbors(n2))

        for nb1 in neighbors1:
            if nb1 in aligned_map:
                nb2 = aligned_map[nb1]
                total_testable += 1
                if nb2 in neighbors2:
                    correct_edges += 1

    edge_correctness = correct_edges / total_testable if total_testable > 0 else 0

    return {
        'num_aligned': n_aligned,
        'coverage_graph1': coverage1,
        'coverage_graph2': coverage2,
        'mean_score': mean_score,
        'edge_correctness': edge_correctness,
        'total_testable_edges': total_testable,
        'correct_edges': correct_edges
    }


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Graph Alignment Example: Semantic Manifold Subspaces")
    print("=" * 60)

    # Example: Two graphs representing similar concepts but from different sources
    # Think of these as two knowledge graphs or two views of the same system

    # Graph 1: A small knowledge graph (e.g., from source A)
    G1 = nx.Graph()
    G1.add_edges_from([
        ("protein_A", "protein_B"),
        ("protein_A", "protein_C"),
        ("protein_B", "protein_C"),
        ("protein_B", "protein_D"),
        ("protein_C", "protein_E"),
        ("protein_D", "protein_E"),
        ("protein_E", "protein_F"),
    ])

    # Graph 2: Similar network from different source (with noise, missing edges, extra nodes)
    G2 = nx.Graph()
    G2.add_edges_from([
        ("gene_A", "gene_B"),
        ("gene_A", "gene_C"),
        ("gene_B", "gene_C"),
        ("gene_B", "gene_D"),
        ("gene_D", "gene_F"),  # Different structure
        ("gene_F", "gene_G"),  # Extra node
    ])

    print(f"\nGraph 1: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
    print(f"Graph 2: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")

    # Define similarity function (this would use your embeddings/features in practice)
    def node_similarity(n1, n2, g1, g2):
        """
        Custom similarity function.
        In practice, you'd use:
        - Embedding similarity (cosine, dot product)
        - Attribute overlap
        - Sequence similarity (for proteins)
        - Semantic similarity (for concepts)
        """
        # Example: crude string matching + degree similarity
        name_sim = 0.8 if n1.split('_')[1] == n2.split('_')[1] else 0.1
        deg_sim = 1.0 / (1.0 + abs(g1.degree[n1] - g2.degree[n2]))
        return 0.7 * name_sim + 0.3 * deg_sim

    # Prepare files
    print("\n" + "=" * 60)
    print("Step 1: Preparing graph files...")
    print("=" * 60)

    net1, net2 = prepare_graphs_for_isorank(G1, G2)
    similarity_file = create_similarity_scores(G1, G2, node_similarity)

    print(f"Network 1: {net1}")
    print(f"Network 2: {net2}")
    print(f"Similarity: {similarity_file}")

    # Run alignment with different parameter settings
    print("\n" + "=" * 60)
    print("Step 2: Running IsoRank alignment...")
    print("=" * 60)

    configs = [
        {"alpha": 0.7, "niter": 1, "name": "Fast (R1, alpha=0.7)"},
        {"alpha": 0.5, "niter": 1, "name": "Topology-focused (R1, alpha=0.5)"},
        {"alpha": 0.9, "niter": 1, "name": "Feature-focused (R1, alpha=0.9)"},
    ]

    for config in configs:
        print(f"\n--- {config['name']} ---")

        alignment = run_isorank_alignment(
            net1, net2, similarity_file,
            alpha=config['alpha'],
            niter=config['niter'],
            npairs=min(G1.number_of_nodes(), G2.number_of_nodes()),
            output_path=f"temp_alignment/alignment_{config['alpha']}.tsv"
        )

        print(f"\nTop 5 alignments:")
        print(alignment.head())

        # Evaluate
        metrics = evaluate_alignment(alignment, G1, G2)

        print(f"\nQuality Metrics:")
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

    print("\n" + "=" * 60)
    print("Alignment complete!")
    print("=" * 60)
