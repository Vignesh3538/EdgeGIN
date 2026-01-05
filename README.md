# <div align="center">**Drug-Drug Interaction Prediction**</div>
<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-PyTorch%20Geometric-orange?style=flat&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-ogbl--ddi-green?style=flat" />
</p>

## ⚠️ **Dataset Overview**
> **The `ogbl-ddi` dataset is a homogeneous, unweighted, undirected graph representing the drug-drug interaction network.**
>
> - **Nodes:** FDA-approved or experimental drugs.
> - **Edges:** Interactions between drugs.
> - **Interpretation:** An edge represents a phenomenon where the joint effect of taking two drugs together is considerably different from the expected effect if the drugs acted independently.

---

## **Prediction Task**

The objective is to predict drug-drug interactions based on existing known interactions.

**Evaluation Metric:** `Hits@K`
The model ranks true drug interactions against non-interacting pairs. Specifically, each true drug interaction is ranked among a set of approximately **100,000 randomly sampled negative drug interactions**. The metric counts the ratio of positive edges ranked at the $K$-th place or above.

$K = 20$ is proposed as a robust threshold based on the OGB initiative's preliminary experiments.

---

## **Dataset Splits**

The `ogbl-ddi` dataset includes three edge splits for **training**, **validation**, and **testing**.

* **Message Passing:**  80% of graph edges in the training data are used for message passing.
* **Supervision:** 20% of train edges are used for train supervision edges; negative edges for training are sampled exclusively based on the training set of edges.

---

## **Edge Features Computation**

To enhance prediction capability, structural features are precomputed from the training set of edges.

**Computed Features:**
* Common Neighbors
* Jaccard Coefficient
* Adamic–Adar Index
* Preferential Attachment
* Resource Allocation Index
* Sørensen Index
* Hub Promoted Index
* Hub Depressed Index

**Implementation Note:**
These features are computed using **chunked processing on a CUDA device**. While this implementation is suitable for graphs up to ~10k nodes, a sparse or sampling-based approach is recommended for larger graphs to avoid quadratic memory and compute overhead.

---

## **GIN Models**

This project explores three variations of Graph Isomorphism Networks (GIN).



### **1. GIN with weighted edge features**
*Location: `/src/EdgeGIN`*

This notebook implements variant of GIN, explicitly incorporating edge features into the message-passing phase by learning edge-specific weights.

#### **Architecture Details**

* **Global Node Embedding Matrix (Layer-Independent):**
    A single learnable node embedding table is initialized using Xavier uniform initialization. It is shared across all GIN layers and serves as the input to the first layer.

* **Precomputed Structural Edge Feature Tensor (Global, Non-Learnable):**
    An 8-dimensional structural feature vector is associated with each unordered node pair $(i, j)$. These are fixed, reused across all layers, and indexed symmetrically using $(\min(i, j), \max(i, j))$ to enforce undirected consistency.

* **Edge-Aware GIN Layers (Layer-Specific):**
    Two `EdgeAwareGINLayer` instances are stacked. Each layer possesses independent parameters, including a residual coefficient $\epsilon$, an edge-weight MLP, and a node-update MLP. No parameters are shared between layers.

* **Edge-Weight Computation (mlp_a):**
    Within each layer, an MLP ($8 \rightarrow 32 \rightarrow 1$) with LayerNorm, ReLU, and Dropout ($0.1$) maps fixed edge features to a scalar weight. These weights are recomputed at every layer and multiplicatively modulate neighbor messages.

* **Message Passing and Aggregation:**
    Incoming neighbor embeddings are scaled by their learned edge weights and aggregated using sum aggregation. While the graph structure (`edge_index`) is shared, the weighting functions are layer-specific.

* **Node Update Function (mlp_phi):**
    Each layer applies a deep MLP with BatchNorm, ReLU, and Dropout ($0.3$) to the residual-augmented aggregation output, preserving embedding dimensionality.

* **Link Prediction Head:**
    A separate MLP operates on the final node embeddings, assigning scores for concatenated pair of node embeddings. This predictor is isolated from message passing.

### **2. Vanilla GIN**
*Location: `/src/VanillaGIN`*

This implementation skips per-layer edge weight computation. It employs a standard stack of **3 GIN layers**.

### **3. GIN with Edge Feature Incorporation at Prediction**
*Location: `/src/GIN_EH`*

- Uses **three standard GIN layers** for node representation learning
- Structural edge features are **not used during message passing**
- Node embeddings are learned independently 
- Final edge scores are computed as an **additive combination** of:
  - a node-embedding interaction score, and
  - a scaled edge-feature score

  **Scoring formulation:**
s(u, v) = f_node(u, v) + alpha * f_edge(u, v)


- `alpha` is a scalar hyperparameter controlling the contribution of structural features
- `f_edge(·)` is learned via a dedicated MLP operating on fixed edge features

---

## **Hyperparameter Selection**

Model hyperparameters are selected based on **validation Hits@20**, including:

- Number of GIN layers
- Hidden dimension size (`hidden_dim`)
- Dropout rates in MLP feed-forward networks
- Choice of nonlinear activation functions
---

## **Model Testing**

- Evaluation and testing are performed under `torch.no_grad()` to disable gradient computation
- Hits@20 is computed using the official **OGB evaluator**
- Metrics are reported separately for **validation** and **test** edge splits
- `/data` contains `.pt` message passing edges files and trained `.pth` model files
