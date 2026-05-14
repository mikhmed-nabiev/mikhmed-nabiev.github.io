# Relational Inductive Biases, Deep Learning, and Graph Networks


Deep learning has delivered remarkable results across image classification, NLP, game playing, and continuous control. Most of those wins were driven by cheap data and cheap compute. But a cluster of harder problems remains stubbornly out of reach: complex scene and language understanding, reasoning about structured relationships, transferring learned skills to new settings, and learning from very little experience.

What do all of these have in common? They require **combinatorial generalization** — the ability to construct new inferences and behaviors from known building blocks, what Humboldt called "the infinite use of finite means." Humans do this naturally by representing the world in terms of *entities* and *relations*, composing familiar pieces to handle novel situations. Current deep learning architectures mostly do not.

The paper argues that the path forward is to embed **relational inductive biases** into deep learning — retaining the flexibility of end-to-end learning while gaining the structured reasoning that combinatorial generalization demands. The vehicle for doing this is the **Graph Network (GN) framework**.

---

## Inductive Biases, Relational and Otherwise

An inductive bias lets a learning algorithm prefer one solution over another independently of the data — it can show up as a prior distribution, a regularization term, or an architectural constraint. A *relational* inductive bias specifically constrains how a model treats relationships and interactions among entities.

Every standard deep learning component already encodes some relational inductive bias:

| Component | Entities | Relations | Key bias | Invariance |
|---|---|---|---|---|
| Fully connected | Units | All-to-all | Weak | — |
| Convolutional | Grid elements | Local | Locality | Spatial translation |
| Recurrent | Timesteps | Sequential | Sequentiality | Time translation |
| Graph network | Nodes | Edges | Arbitrary | Node/edge permutation |

The first three components impose either no structure (fully connected) or a rigid, domain-specific structure (CNNs assume spatial locality; RNNs assume sequential order). None of them handle arbitrary relational structure — which is exactly what's needed for reasoning about physical systems, molecules, social networks, or any problem where what matters is *who is connected to whom*, not a fixed spatial or temporal layout.

---

## From Sets to Graphs

Before introducing the full framework, it helps to see how set-based and graph-based computations differ from standard layers.

**Deep Sets** handle unordered collections by applying a symmetric aggregation over all pairs:

$$x'_i = f\!\left(x_i,\; \sum_j g(x_i, x_j)\right)$$

The key property is permutation invariance: the output doesn't change if you reorder the inputs.

**Graphs** extend this to *sparse* pairwise structure. Rather than aggregating over all pairs, each node only aggregates over its neighborhood $\delta(i)$:

$$x'_i = f\!\left(x_i,\; \sum_{j \in \delta(i)} g(x_i, x_j)\right)$$

The sparsity is the point — it encodes which entities directly interact, and the structure of those interactions is part of the input, not baked into the architecture.

---

## The GN Framework

### Graphs as Data Structures

In the GN framework, a graph is a triple $G = (\mathbf{u}, V, E)$:

- $\mathbf{u}$ — a **global attribute** shared across the whole graph (e.g., a gravitational constant or energy value).
- $V = \{\mathbf{v}_i\}$ — **node attributes** (e.g., position, velocity, and mass of each object).
- $E = \{(\mathbf{e}_k, r_k, s_k)\}$ — **edge attributes** with a receiver index $r_k$ and sender index $s_k$ (e.g., spring constants between pairs of objects).

A GN block is a *graph-to-graph* module: it takes a graph $G$ as input and returns an updated graph $G'$.

### Update and Aggregation Functions

A GN block contains three **update functions** $\phi$ and three **aggregation functions** $\rho$:

$$\mathbf{e}'_k = \phi^e(\mathbf{e}_k,\, \mathbf{v}_{r_k},\, \mathbf{v}_{s_k},\, \mathbf{u})$$
$$\mathbf{v}'_i = \phi^v(\bar{\mathbf{e}}'_i,\, \mathbf{v}_i,\, \mathbf{u})$$
$$\mathbf{u}' = \phi^u(\bar{\mathbf{e}}',\, \bar{\mathbf{v}}',\, \mathbf{u})$$

with aggregations:

$$\bar{\mathbf{e}}'_i = \rho^{e \to v}(E'_i), \qquad \bar{\mathbf{e}}' = \rho^{e \to u}(E'), \qquad \bar{\mathbf{v}}' = \rho^{v \to u}(V')$$

Computation proceeds in order: edges first, then nodes, then the global attribute. The aggregation functions $\rho$ must be permutation invariant and handle variable-size inputs — elementwise sum, mean, or max all work.

### Neural Network Implementation

In practice, $\phi^e$, $\phi^v$, and $\phi^u$ are implemented as neural networks (typically MLPs for vector attributes, CNNs for tensor attributes, or RNNs when a hidden state is needed). The aggregations use elementwise summation:

$$\rho^{e \to v}(E'_i) = \sum_{\{k:\, r_k = i\}} \mathbf{e}'_k, \qquad \rho^{v \to u}(V') = \sum_i \mathbf{v}'_i, \qquad \rho^{e \to u}(E') = \sum_k \mathbf{e}'_k$$

---

## GN as a Unifying Framework

One of the more satisfying results in the paper is that many well-known architectures turn out to be special cases of this framework, obtained by zeroing out or simplifying certain components:

- **Message Passing Neural Networks (MPNN)**: $\phi^e$ ignores the global attribute; a readout function replaces $\phi^u$.
- **Non-local Neural Networks / Self-Attention**: $\phi^e$ factors into a scalar attention weight $\alpha^e(\mathbf{v}_{r_k}, \mathbf{v}_{s_k})$ and a value $\beta^e(\mathbf{v}_{s_k})$; only node outputs are produced.
- **Relation Networks**: the node update is bypassed; the global attribute is predicted directly from pooled edge representations.
- **Deep Sets**: the edge update is bypassed; the global attribute is predicted directly from pooled node representations.

The GN block is not a new architecture so much as a lingua franca for a whole family of relational architectures.

---

## Composable Multi-Block Architectures

GN blocks have a graph-to-graph interface, which makes them naturally composable:

$$G' = \mathrm{GN}_2(\mathrm{GN}_1(G))$$

Stacked blocks with *shared* weights are analogous to an unrolled RNN — information propagates $m$ hops after $m$ steps. Stacked blocks with *different* weights behave more like the layers of a CNN.

The most common design pattern is **encode-process-decode**:
1. $\mathrm{GN}_\mathrm{enc}$ maps the input graph to a latent graph $G_0$.
2. A shared $\mathrm{GN}_\mathrm{core}$ is applied $M$ times.
3. $\mathrm{GN}_\mathrm{dec}$ maps the result to the output graph.

---

## Why GNs Support Combinatorial Generalization

GNs encode three relational inductive biases that directly support out-of-distribution generalization:

**Arbitrary relational structure.** The graph structure itself — which nodes connect to which — is part of the input, not fixed by the architecture. Edges define who interacts; the absence of an edge means no direct influence.

**Permutation invariance.** Entities and relations are represented as sets, so the output is invariant to node and edge ordering. There's no privileged "first" node.

**Function reuse across entities.** The same $\phi^e$ and $\phi^v$ are applied to every edge and node. A single trained GN can operate on graphs of different sizes and connectivity patterns.

That last point is what makes combinatorial generalization tractable: a model that applies the same rule to each entity can, in principle, handle any number of entities at test time.

The empirical evidence is encouraging. GNs trained for one-step physical simulation have been applied for thousands of steps without retraining, and transferred zero-shot to systems with twice or half as many entities. Forward models for multi-joint control generalize to agents with unseen limb counts. Learned solvers for combinatorial optimization and Boolean SAT retain performance well outside their training distribution.

---

## Limitations and Open Questions

The framework is not without gaps.

On the theoretical side, learned message-passing cannot distinguish all non-isomorphic graphs — there exist structurally different graphs that any GN will process identically. Recursion, conditional branching, and variable-length iteration also cannot be natively expressed as static graph structure.

The harder open problem is **graph construction**: given raw perceptual inputs like images or text, how do you extract a meaningful sparse graph in the first place? The paper treats the input graph as given, but in most real applications it has to be inferred or approximated, and this remains largely unsolved. Related is the question of **adaptive structure**: how to add or remove nodes and edges dynamically during computation — necessary for modeling processes like object fracturing or cell division.

---

## Takeaway

The core argument is tight: combinatorial generalization is the right target for AI, structured representations are the right tool, and graph networks are a principled way to incorporate that structure into deep learning without giving up end-to-end trainability. The framework is flexible enough to subsume most prior work on graph neural networks, and its relational inductive biases provide a concrete account of *why* these models generalize in the ways they do.

---

*Based on: Battaglia et al. "Relational Inductive Biases, Deep Learning, and Graph Networks." arXiv:1806.01261 (2018).*
