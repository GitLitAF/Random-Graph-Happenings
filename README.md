# Graph Alignment for Semantic Manifold Subspaces

This repository contains tools and examples for **probabilistic graph alignment** - aligning graphs from disparate sources that represent different views of the same underlying semantic space.

## Problem Statement

You have two graphs that:
- Represent the same underlying concept/system (semantic manifold)
- Come from different sources (e.g., different databases, experiments, models)
- May have **partial overlap** (from single node to full alignment)
- Need **probabilistic/similarity-based** alignment (not exact matching)

This is analogous to **local sequence alignment** (Smith-Waterman) in bioinformatics, but for graphs.

## Approach: IsoRank Algorithm

We use **IsoRank**, a spectral network alignment algorithm that:

### Core Idea
- Combines **topology** (graph structure) with **node features** (attributes/embeddings)
- Computes a **similarity matrix** R where R[i,j] = probability that node i aligns to node j
- Uses **iterative refinement** inspired by PageRank

### Algorithm

```
R = (1-α) * (P2 @ R @ P1^T) + α * E
```

Where:
- **E**: Initial node similarity matrix (your prior knowledge - like BLAST scores)
- **P1, P2**: Normalized adjacency matrices (graph topology)
- **α**: Balance parameter (0=topology only, 1=features only, ~0.7 recommended)
- **R**: Output similarity matrix

### Quality Metrics
- **Coverage**: Fraction of nodes aligned
- **Edge Correctness**: Fraction of edges preserved by alignment
- **Mean Similarity**: Average alignment confidence

## Installation

### Option 1: Install the netalign package

```bash
cd approximate_ISORANK
pip install -e .
```

Dependencies: numpy, scipy, pandas, networkx, matplotlib, scikit-learn, torch

### Option 2: Use standalone implementation

Just use `graph_alignment_example.py` - only requires networkx and numpy.

## Usage Examples

### Quick Start: Standalone Implementation

```python
import networkx as nx
from graph_alignment_example import (
    create_node_similarity_matrix,
    compute_isorank_alignment,
    greedy_alignment,
    compute_alignment_quality
)

# Create two graphs
G1 = nx.karate_club_graph()
G2 = nx.karate_club_graph()
# ... perturb G2 somehow ...

# Create similarity matrix (customize this for your domain!)
E = create_node_similarity_matrix(G1, G2)

# Compute alignment
R, info = compute_isorank_alignment(G1, G2, E, alpha=0.7, max_iterations=20)

# Get best matches
alignments = greedy_alignment(R, num_pairs=20, threshold=0.1)

# Evaluate quality
metrics = compute_alignment_quality(alignments, G1, G2)
print(metrics)
```

### Advanced: Using netalign Package

```python
from isorank_advanced_example import (
    prepare_graphs_for_isorank,
    create_similarity_scores,
    run_isorank_alignment,
    evaluate_alignment
)

# Define custom node similarity
def my_similarity(n1, n2, g1, g2):
    # Use embeddings, attributes, etc.
    embedding1 = g1.nodes[n1].get('embedding')
    embedding2 = g2.nodes[n2].get('embedding')
    return cosine_similarity(embedding1, embedding2)

# Prepare files
net1, net2 = prepare_graphs_for_isorank(G1, G2)
sim_file = create_similarity_scores(G1, G2, my_similarity)

# Run alignment
alignment = run_isorank_alignment(
    net1, net2, sim_file,
    alpha=0.7,  # Balance topology vs features
    niter=1,     # R1 approximation (fast)
    npairs=100   # Top 100 matches
)

# Evaluate
metrics = evaluate_alignment(alignment, G1, G2)
```

## Key Parameters

### Alpha (α) - Balance Parameter
- **0.0-0.3**: Topology-driven (ignore node features, focus on graph structure)
- **0.5-0.7**: **Recommended** - balanced
- **0.8-1.0**: Feature-driven (ignore topology, focus on node similarity)

### Iterations (niter) - Accuracy vs Speed
- **0**: R0 approximation (fastest, least accurate)
- **1**: R1 approximation (**recommended** - fast & accurate)
- **10+**: Full IsoRank (slower, marginal improvement)

### Similarity Function
The most important customization! This encodes your domain knowledge:

```python
def similarity(node1_attrs, node2_attrs):
    """
    Examples:
    - Embedding similarity: cosine(embed1, embed2)
    - Sequence similarity: BLAST score (for proteins)
    - Attribute overlap: Jaccard(attrs1, attrs2)
    - Semantic similarity: WordNet, BERT, etc.
    """
    return score  # 0.0 to 1.0
```

## File Structure

```
.
├── README.md                          # This file
├── graph_alignment_example.py         # Standalone implementation
├── isorank_advanced_example.py        # Using netalign package
└── approximate_ISORANK/               # IsoRank package
    ├── netalign/
    │   ├── approx_isorank/
    │   │   ├── isorank_compute.py     # Core algorithm
    │   │   ├── io_utils.py            # Data loading
    │   │   └── pair_evaluations.py    # Metrics
    │   └── duomundo/                  # Alternative method
    ├── notebooks/                      # Jupyter examples
    └── data/                           # Example datasets
```

## Comparison with Other Approaches

| Method | Type | Use Case | Speed | Accuracy |
|--------|------|----------|-------|----------|
| **IsoRank** | Spectral | General graphs | Fast (O(n²)) | High |
| Graph Isomorphism | Exact | Identical graphs | NP-hard | Perfect (when exists) |
| GED (Graph Edit Distance) | Edit-based | Small graphs | Slow (NP-hard) | Exact |
| REGAL | Embedding | Large graphs | Very fast | Good |
| VF2 (subgraph) | Exact | Pattern matching | Fast (small patterns) | Perfect (when exists) |

IsoRank is ideal for:
- ✅ Graphs from different sources representing the same thing
- ✅ Partial overlaps (local alignment)
- ✅ Incorporating node features/embeddings
- ✅ Scalability (millions of nodes)

## Biological Context (Original Use Case)

IsoRank was designed for **protein-protein interaction (PPI) network alignment**:

- **Graph 1**: PPI network from yeast
- **Graph 2**: PPI network from human
- **Node similarity (E)**: BLAST sequence similarity scores
- **Goal**: Find functionally equivalent proteins (orthologs)

Your use case is analogous:
- **Graph 1**: Knowledge graph from source A
- **Graph 2**: Knowledge graph from source B
- **Node similarity (E)**: Embedding similarity, attribute overlap, etc.
- **Goal**: Find corresponding concepts/entities

## Advanced Topics

### Handling Spacer Nodes (Gaps)
The alignment naturally handles "gaps" through:
1. **Greedy assignment**: Nodes with low similarity won't be aligned
2. **Threshold**: Set minimum similarity in `greedy_alignment(threshold=0.X)`
3. **Coverage metric**: Track what fraction remains unaligned

### Multi-Graph Alignment
For >2 graphs, see:
- **IsoRankN** (in approximate_ISORANK repo)
- **Progressive alignment**: Align pairwise, then merge

### Weighted Graphs
Modify adjacency matrix to include edge weights:
```python
A = nx.to_numpy_array(G, weight='weight')
```

### Directed Graphs
Use directed adjacency:
```python
# Separate in/out normalization
A_out = A / A.sum(axis=1, keepdims=True)
A_in = A / A.sum(axis=0, keepdims=True)
```

## References

**Original IsoRank Paper:**
- Singh, R., Xu, J., & Berger, B. (2008). *Global alignment of multiple protein interaction networks with application to functional orthology detection*. PNAS, 105(35), 12763-12768.

**IsoRank-N (Multiple networks):**
- Liao, C. S., Lu, K., Baym, M., Singh, R., & Berger, B. (2009). *IsoRankN: spectral methods for global alignment of multiple protein networks*. Bioinformatics, 25(12), i253-i258.

**Fast Approximate IsoRank:**
- Devkota, K., Cowen, L. J., Blumer, A., & Hu, X. (2023). *Fast Approximate IsoRank for Scalable Global Alignment of Biological Networks*. bioRxiv.

**Package Repository:**
- https://github.com/kap-devkota/approximate_ISORANK

## License

This repository: MIT License

approximate_ISORANK package: MIT License (see package LICENSE file)

## Citation

If you use this in research, please cite the IsoRank papers above.
