# Methodology: Design Decisions and Scientific Rationale

This document explains the key design decisions in the `feature_graph` library — why we chose specific approaches over alternatives, where the methodology has known limitations, and how researchers should interpret results.

---

## 1. Why Clamping Over Other Causal Estimators

We implement **feature clamping** as the primary interaction measurement. The intervention works by modifying the residual stream at the source feature's layer:

$$\hat{\mathbf{h}}^{(l)}(\alpha) = \mathbf{h}^{(l)} + (\alpha - f^{(l)}_i) \cdot \mathbf{d}^{(l)}_i$$

then measuring the downstream target feature activation $f^{(l')}_j$.

### Alternatives considered and why we didn't use them as primary:

**Gradient / Jacobian**: We provide this as a fast secondary method. Gradients give the local linear approximation $\partial f_j / \partial f_i$, which is accurate when the interaction is approximately linear and the perturbation is small. But SAE features go through GeLU/SiLU nonlinearities in MLP layers, which means the actual response to a perturbation can be very different from the gradient — especially near the nonlinearity's transition region. Gradients also don't distinguish "the gradient is nonzero but the target feature stays at zero because of the ReLU threshold in the SAE encoder" from "the target feature actually changes." Clamping measures the *actual* response.

**Path patching**: Decomposes the effect along specific computational paths (through specific heads/MLPs). Excellent for understanding *mechanism* but requires enumerating paths, which is $O(\text{exponential})$ in the number of layers between source and target. We recommend path patching for deep-diving into specific edges *after* the graph is built, not for constructing the graph.

**Activation patching (mean ablation)**: Replaces the full residual stream at a layer with the dataset mean. This is too coarse — it ablates *all* features at that layer, not just the one we care about. Feature clamping is a surgical version that modifies exactly one feature's contribution.

**Crosscoder weight analysis**: Looking at decoder-encoder alignment across layers. This captures only the *direct residual stream path* — it misses attention-mediated and MLP-mediated interactions entirely. We use it as a pruning filter (candidate identification), not as the measurement.

### Known limitations of clamping:

1. **Superposition leakage**: When we add $\Delta \alpha \cdot \mathbf{d}_i$ to the residual stream, this perturbation has nonzero projections onto *other* feature directions (because features are not perfectly orthogonal in superposition). This means some of the measured "interaction" between $f_i$ and $f_j$ is actually mediated by these incidental projections onto other features. Mitigation: we use relatively small perturbations (between the 10th and 90th percentile of the feature's activation distribution) to stay in a regime where the perturbation is small relative to the full residual stream norm.

2. **Nonlinearity of the response**: The interaction strength $I$ is a *linear* summary of a potentially nonlinear response curve $f_j(\alpha)$. If the response is strongly nonlinear (e.g., threshold effects), our two-point measurement (high vs. low clamp) gives the average slope, which may not represent the interaction well at all operating points. For important edges, we recommend computing the full response curve by sweeping $\alpha$.

3. **Context dependence**: The interaction strength varies across inputs (this is the source of "gating" interactions). Our reported $I$ is the *mean* across inputs, with the *variance* used to detect gating. This means an edge labeled "excitatory" might actually be inhibitory on some inputs — the label describes the average behavior.

---

## 2. Statistical Testing

### Why we use t-tests with FDR correction (not permutation tests)

The research document proposes permutation tests for the null distribution. After implementation testing, we found that:

- **Permutation tests are cleaner in principle** but require shuffling feature activations, which breaks the temporal structure of the residual stream in ways that make the null distribution unrealistically broad.
- **A t-test on the per-input interaction strengths** is computationally cheaper, works well when we have ≥20 samples per pair, and gives calibrated p-values when the distribution of interaction strengths is approximately normal (which it is, by CLT, when aggregating across many token positions per input).

We use **Benjamini-Hochberg FDR correction** rather than Bonferroni because:
- We're testing thousands of pairs, and Bonferroni is too conservative (it controls the family-wise error rate, but we care about the false discovery rate).
- BH-FDR at level 0.01 means approximately 1% of our reported significant edges are false positives — acceptable for a discovery-oriented analysis.

### Gating detection

We detect gating interactions by computing the **coefficient of variation** (CV) of the interaction strength across inputs:

$$CV = \frac{\text{std}(I)}{\text{mean}(|I|)}$$

A high CV (above `gating_variance_threshold`, default 2.0) indicates that the interaction strength is highly context-dependent — sometimes strong, sometimes weak, sometimes changing sign. This is the signature of a modulatory/gating interaction where a third feature determines whether the source-target interaction is active.

---

## 3. Graph Construction: What the edges mean

Each edge in the interaction graph represents a **structural causal relationship** aggregated across many inputs. This is fundamentally different from:

- **Anthropic's attribution graphs**: Per-prompt, gradient-based. Our edges are averaged across inputs and measured via interventions.
- **Correlation**: We measure what happens when we *change* the source, not whether source and target co-vary.
- **Granger causality**: We're not doing time-series analysis; the "causal" direction comes from the layer ordering of the transformer.

### Edge types

| Type | Criterion | Meaning |
|------|-----------|---------|
| **Excitatory** | $I > 0$, $CV < \tau$ | Increasing $f_i$ increases $f_j$, consistently across inputs |
| **Inhibitory** | $I < 0$, $CV < \tau$ | Increasing $f_i$ decreases $f_j$, consistently across inputs |
| **Gating** | $CV > \tau$ | The effect of $f_i$ on $f_j$ depends strongly on context (other features) |

### What "strength" means

The edge strength $I$ has units of "target feature activation change per unit source feature activation change." A strength of 0.5 means: if you increase the source feature by 1 unit (1 standard deviation of its activation), the target feature increases by 0.5 units on average. This is directly useful for predicting the effect of steering interventions.

---

## 4. Pruning Pipeline: Why we don't test all pairs

With $D \sim 24{,}000$ features per layer and $L = 12$ layers, testing all pairs would require $\sim 10^{10}$ intervention experiments. Our pruning pipeline reduces this to $\sim 10^4$–$10^5$ testable pairs:

1. **Importance filter (top-K)**: We only study the most important features. "Importance" = activation frequency × causal effect on model output. This captures features that are both frequently used and meaningfully influence the model's predictions.

2. **Co-activation filter**: Features that never co-activate on the same inputs cannot have observationally relevant interactions. We keep pairs with high PMI (positive co-activation) and pairs with anomalously low co-activation (candidate inhibition).

3. **Decoder alignment filter**: For adjacent layers, the direct residual-stream interaction between features is proportional to the dot product of the source decoder direction and target encoder direction. Pairs with near-zero alignment have no direct connection (though they may interact via attention/MLP paths).

4. **Layer locality**: Most interactions are local ($|l' - l| \leq 3$). Long-range interactions exist but are typically mediated by chains of shorter-range interactions, which the local graph captures as multi-hop paths.

### What we miss

This pruning is aggressive by design (for tractability), which means we miss:

- **Rare but strong interactions**: If two features almost never co-activate but have a powerful causal relationship when they do, our co-activation filter will discard them. This is a known limitation; increase `n_tokens` and lower `coactivation_threshold` to mitigate.
- **Long-range skip interactions**: A feature at layer 2 that directly influences a feature at layer 11 (bypassing all intermediate layers) will be missed if `layer_window < 9`. Increase the window for long-range analysis, but expect quadratic scaling.
- **Low-importance features**: Features not in the top-K are excluded entirely. A feature with low activation frequency but critical safety relevance could be missed. For safety-critical analyses, include specific features of interest in the candidate set manually.

---

## 5. How to interpret the graph

### The graph is a model of the model

The interaction graph is a *compressed representation* of the transformer's computational structure. It is not "the truth" — it is a useful approximation that:

- **Preserves causal direction**: Edges go from earlier layers to later layers, respecting the transformer's computational order.
- **Summarizes nonlinear dynamics linearly**: Each edge is a linear interaction strength, summarizing a potentially nonlinear relationship.
- **Aggregates across contexts**: Each edge represents the *average* interaction, smoothing over context-dependent variation (except for gating edges, which flag high variation).

### When to trust the graph

- **Hub identification**: Robust to noise. A feature with 50 edges is genuinely central to the computation.
- **Strong excitatory/inhibitory edges**: Edges with $|I| > 0.3$ and $p < 0.001$ are reliable.
- **Community structure**: The Louvain communities are meaningful as "features that tend to work together."

### When to be cautious

- **Weak edges**: Edges near the significance threshold may be false positives (even with FDR correction).
- **Gating interactions**: The gating label means "we detected high variance" — the actual gating mechanism requires further investigation (conditioning on third features).
- **Cascade predictions**: Multi-hop cascade predictions compound errors. Trust 1-hop predictions much more than 3-hop predictions.
- **Absence of edges**: The absence of an edge does NOT mean two features don't interact — it means the interaction wasn't detected with our method and sample size. Absence of evidence ≠ evidence of absence.
