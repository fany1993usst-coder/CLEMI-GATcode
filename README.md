

Contrastive Learning Enhanced Multi-level Interest-aware Graph Attention Networks for Session-based Recommendation

Overview
1. CLEMI-GAT is a session-based recommendation model that captures user interests at multiple levels by constructing:

2. Multi-level interest-aware hypergraph – models atomic, transitional, and composite interests via hyperedges of varying cardinalities.

3. Intermediary interaction graph – introduces a virtual node to directly capture long-distance dependencies between items.

4. Gated fusion network – adaptively combines the two channel representations.

5. RENorm – separates in-session and out-of-session items to balance repeat and exploratory behaviors.

6. Contrastive learning – employs InfoNCE with in-batch negatives to alleviate data sparsity.

Results
Experiments on three public datasets (Diginetica, Yoochoose1/64, Yoochoose1/4) show that CLEMI-GAT consistently outperforms state-of-the-art baselines in terms of P@20 and MRR@20. Detailed performance comparisons and ablation studies are provided in the paper.
