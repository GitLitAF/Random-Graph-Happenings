# Quick Start Guide

## What We Built

A complete graph alignment solution for aligning graphs from disparate sources representing the same underlying semantic space - perfect for your bioinformatics-inspired use case!

## Installation

### Quick (Standalone)
Just need numpy and networkx:
```bash
pip install numpy networkx
python graph_alignment_example.py
```

### Full (with netalign package)
```bash
pip install -r requirements.txt
cd approximate_ISORANK
pip install -e .
```

## Simplest Example

```python
import networkx as nx
from graph_alignment_example import (
    create_node_similarity_matrix,
    compute_isorank_alignment,
    greedy_alignment
)

# Your two graphs
G1 = nx.karate_club_graph()  # Graph from source A
G2 = nx.karate_club_graph()  # Graph from source B
# ... modify G2 to simulate differences ...

# Step 1: Create similarity matrix (customize for your domain!)
E = create_node_similarity_matrix(G1, G2)

# Step 2: Compute alignment
R, info = compute_isorank_alignment(
    G1, G2, E,
    alpha=0.7,  # Balance topology (0) vs features (1)
    max_iterations=20
)

# Step 3: Get best alignments
alignments = greedy_alignment(
    R,
    num_pairs=20,      # Top 20 matches
    threshold=0.1      # Minimum confidence
)

# Results: [(node1_idx, node2_idx, similarity_score), ...]
for i, j, score in alignments:
    print(f"Node {i} aligns to Node {j} with confidence {score:.3f}")
```

## Key Customization: Similarity Function

The most important part for your use case! Define how nodes should be compared:

```python
def my_similarity(node1_attrs, node2_attrs):
    """
    For semantic manifold alignment, you might use:
    - Embedding similarity (cosine)
    - Attribute overlap (Jaccard)
    - Sequence similarity (BLAST for proteins)
    - Semantic similarity (WordNet, BERT, etc.)
    """
    # Example: cosine similarity of embeddings
    emb1 = node1_attrs.get('embedding')
    emb2 = node2_attrs.get('embedding')
    return cosine_similarity(emb1, emb2)

# Use it:
E = create_node_similarity_matrix(G1, G2, similarity_func=my_similarity)
```

## Understanding the Parameters

### Alpha (α) - Most Important!
- **0.0-0.3**: Trust topology over features (good when embeddings are noisy)
- **0.5-0.7**: **Recommended** - balanced approach
- **0.8-1.0**: Trust features over topology (good when structure is noisy)

### Threshold
- Set minimum alignment confidence
- Higher = fewer but more confident alignments
- Lower = more alignments but some may be weak

### Iterations (niter)
- **1**: Fast approximation (R1) - **recommended**
- **10+**: More accurate but slower

## What Makes This Different from Graph Matching

❌ **Graph Isomorphism**: Requires exact structural match (won't work for your case)
❌ **Subgraph Matching**: Finds exact patterns (not probabilistic)
❌ **Graph Edit Distance**: Expensive for large graphs

✅ **IsoRank Alignment**:
- Probabilistic similarity-based
- Handles partial overlaps (like local alignment in sequences)
- Combines structure + features
- Fast and scalable
- Outputs confidence scores

## Next Steps

1. **Test with your data**: Replace the example graphs with your actual graphs
2. **Customize similarity**: Implement your domain-specific node similarity
3. **Tune alpha**: Experiment with different balances of topology vs features
4. **Set thresholds**: Find the right confidence cutoff for your use case
5. **Evaluate quality**: Use the provided metrics to assess alignment quality

## Files to Explore

- `graph_alignment_example.py` - Start here! Standalone implementation with examples
- `isorank_advanced_example.py` - Advanced usage with the netalign package
- `README.md` - Comprehensive documentation
- `approximate_ISORANK/notebooks/` - Jupyter notebook examples

## Common Use Cases

### Knowledge Graph Alignment
Aligning entities from different knowledge bases (e.g., DBpedia + Wikidata)

### Biological Network Alignment
Finding protein orthologs across species (original IsoRank use case)

### Multi-Modal Representations
Aligning different representations of the same underlying semantic space

### Schema Matching
Aligning database schemas or ontologies

## Getting Help

See `README.md` for:
- Detailed algorithm explanation
- Parameter tuning guide
- Quality metrics interpretation
- Advanced topics (directed graphs, weighted edges, etc.)
- References and citations
