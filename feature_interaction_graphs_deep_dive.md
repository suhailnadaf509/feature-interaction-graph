# The Feature Interaction Graph: Mapping Compositional Computation in SAE Feature Space

## A Deep Technical Description

*Author: Mohammed Suhail B Nadaf*
*Date: February 28, 2026*

---

## 1. The Core Problem: We Have a Parts List but No Wiring Diagram

The mechanistic interpretability community has spent the past three years building an increasingly impressive **parts catalog** of neural network internals. Sparse autoencoders have given us dictionaries of thousands of interpretable features per layer — "Golden Gate Bridge," "legal terminology," "Python function definitions," "deceptive intent." Neuronpedia lets you browse them. Anthropic's Gemma Scope provides pre-trained SAEs across every layer of Gemma-2. The catalog grows daily.

And yet: **we have almost no understanding of how these features interact with each other.**

This is not a minor gap. It is the gap between knowing the names of every protein in a cell and understanding the metabolic pathways that keep the cell alive. Between having a dictionary and having a grammar. Between identifying every transistor on a chip and having the circuit schematic.

Consider what we currently *cannot* answer:

1. **How do features compose across layers?** When the model processes "The Eiffel Tower is in Paris," does a "European landmark" feature in layer 8 *cause* a "capital city" feature in layer 14 to activate, which in turn *causes* a "France" feature in layer 20 to fire? Or do all three activate independently from the input? The difference between causal chaining and independent activation is the difference between understanding a computation and cataloging correlations.

2. **Which features inhibit each other?** When a "refusal" feature fires, does it *suppress* a "helpful response generation" feature? When a "formal register" feature is active, does it *gate off* a "casual slang" feature? Inhibitory relationships are computationally essential — they implement selection, competition, and routing — but we have zero systematic knowledge of them in SAE feature space.

3. **What does the feature-level "circuit" for a complex behavior look like?** We can describe the IOI circuit in terms of attention heads (S-inhibition heads, name-mover heads, etc.). But we cannot describe it — or any behavior — in terms of SAE features and their causal relationships. The head-level circuit and the feature-level circuit are two descriptions of the same computation at different resolutions, and we only have the coarser one.

4. **Are there "hub" features that participate in many computations?** Network science tells us that real-world graphs are typically not random — they have structure: hubs, clusters, bottlenecks, hierarchies. Does the feature interaction graph have similar structure? Are there features that act as critical routing nodes, participating in dozens of downstream computations? If so, these hub features are simultaneously the most important to understand (they affect everything) and the most dangerous to steer (modifying them cascades unpredictably).

5. **When we steer on a single feature, what cascades?** Current activation steering modifies one feature (or direction) at a time and hopes for the best. But if that feature excites five others and inhibits three, the downstream effects are a function of the entire local interaction graph, not just the target feature. Without the interaction graph, steering is surgery without anatomy.

---

## 2. Why This Is the Natural Next Step for SAE-Based Interpretability

### 2.1 The State of the Art and Its Ceiling

As of February 2026, SAE-based interpretability has converged on a standard pipeline:

1. **Train SAEs** on residual stream activations at each layer (or on MLP outputs via transcoders).
2. **Identify features** by examining their maximum-activating examples, auto-interpretability scores (LLM-generated descriptions), and activation patterns.
3. **Attribute model behavior to features** via activation patching, attribution graphs, or direct feature contribution analysis (projecting feature decoder directions onto the unembedding/output direction).

This pipeline has been enormously productive. Anthropic's attribution graphs trace token-level computation through features on individual prompts. Gemma Scope provides a pre-computed parts catalog. Neuronpedia makes features browsable. The community has identified features corresponding to concepts ranging from "DNA sequences" to "expressions of uncertainty" to "code comments in French."

But the pipeline has a hard ceiling: **it treats features as atomic, independent units.** Each feature is studied, described, and cataloged in isolation. The attribution graph traces a *specific prompt* through features — it tells you "on this input, feature A at layer 5 contributed to feature B at layer 12." It does not tell you whether this A→B connection is a *structural property of the model* (true across many inputs) or an *incidental co-activation* specific to that prompt.

The distinction matters enormously. Structural connections are the model's "wiring" — they define its computational architecture. Incidental co-activations are just input-driven correlations. Conflating the two is like conflating the fact that two genes happen to be expressed in the same tissue with the claim that one gene regulates the other.

### 2.2 The Analogy to Gene Regulatory Networks

The best analogy for where we are comes from molecular biology. By the late 1990s, the Human Genome Project had cataloged ~20,000 genes. This was an extraordinary achievement, but it quickly became clear that knowing the parts list was insufficient for understanding biological function. The breakthrough came from mapping **gene regulatory networks** — which genes activate which others, which repress which others, which form feedback loops, and how these interactions give rise to cell types, developmental programs, and disease states.

SAE features are our "genes." We're cataloging them with increasing completeness and accuracy. But without the regulatory network — the interaction graph — we cannot understand how they give rise to *computation*. We have the genome. We need the interactome.

### 2.3 What Anthropic's Attribution Graphs Do and Don't Do

Anthropic's attribution graphs (the "Biology of a Large Language Model" work) are the closest existing thing to a feature interaction map. They deserve careful analysis because understanding what they achieve — and where they stop — is essential for positioning this project.

**What attribution graphs do:**
- Trace the causal flow of information through SAE features *for a specific prompt*.
- Show which features at layer $l$ contribute to which features at layer $l+k$, mediated by attention and MLP computation.
- Produce beautiful, interpretable visualizations of per-prompt computational pathways.

**What attribution graphs do NOT do:**
- **They are prompt-specific, not structural.** The graph for "The Eiffel Tower is in Paris" is different from the graph for "The Colosseum is in Rome." The question "does a 'European landmark' feature generally excite a 'capital city' feature?" is not answered by any single attribution graph — it requires aggregating across many inputs, which Anthropic's published work does not do.
- **They do not characterize interaction types.** An attribution graph shows that feature A "influenced" feature B, but it doesn't distinguish excitation (A increases B) from inhibition (A decreases B) from gating (A modulates B's sensitivity to other inputs). These are fundamentally different computational relationships.
- **They do not provide a global network topology.** Because each attribution graph is prompt-specific, there is no way to compute global graph statistics — degree distributions, clustering coefficients, hub identification, community structure — that would reveal the *architectural* principles of feature organization.
- **They do not capture feature-to-feature causal strength quantitatively.** The edges in attribution graphs represent gradient-based attributions, which are local linear approximations. They don't tell you "if I clamp feature A to its maximum activation, feature B's activation increases by 0.7 standard deviations" — which is what you need for predicting the effects of interventions.

The Feature Interaction Graph project fills exactly these gaps: structural (not prompt-specific), typed (excitation/inhibition/gating), global (network-level statistics), and quantitatively causal (measured via interventions, not just gradients).

---

## 3. Formal Framework: Defining Feature Interactions

Before designing experiments, we need a rigorous formalism. The core challenge is: **what does it mean for SAE feature $f_i$ at layer $l$ to "interact with" SAE feature $f_j$ at layer $l'$?**

### 3.1 Setup and Notation

Consider a transformer with $L$ layers and a trained SAE at each layer. Let:

- $\mathbf{h}^{(l)} \in \mathbb{R}^d$ be the residual stream at layer $l$ (at a given token position; we'll suppress the position index for clarity).
- $E^{(l)}: \mathbb{R}^d \to \mathbb{R}^{D_l}$ be the SAE encoder at layer $l$, producing feature activations $\mathbf{f}^{(l)} = E^{(l)}(\mathbf{h}^{(l)})$ where $D_l \gg d$ is the dictionary size.
- $D^{(l)}: \mathbb{R}^{D_l} \to \mathbb{R}^d$ be the SAE decoder, with column vectors $\mathbf{d}^{(l)}_i$ being the decoder directions (feature directions in activation space).
- $f^{(l)}_i$ denote the scalar activation of feature $i$ at layer $l$.

The residual stream is approximately reconstructed as:

$$\mathbf{h}^{(l)} \approx \sum_i f^{(l)}_i \cdot \mathbf{d}^{(l)}_i + \mathbf{b}^{(l)}_{\text{dec}}$$

where $\mathbf{b}^{(l)}_{\text{dec}}$ is the decoder bias (absorbing the mean).

### 3.2 Three Types of Feature Interaction

We define three operationally distinct types of interaction between features at different layers:

#### 3.2.1 Excitation

Feature $f^{(l)}_i$ **excites** feature $f^{(l')}_j$ (where $l' > l$) if artificially increasing $f^{(l)}_i$ causes $f^{(l')}_j$ to increase.

Formally, let $\hat{\mathbf{h}}^{(l)}(\alpha)$ denote the residual stream at layer $l$ after clamping feature $i$ to activation level $\alpha$:

$$\hat{\mathbf{h}}^{(l)}(\alpha) = \mathbf{h}^{(l)} + (\alpha - f^{(l)}_i) \cdot \mathbf{d}^{(l)}_i$$

Let $\hat{f}^{(l')}_j(\alpha)$ be the resulting activation of feature $j$ at layer $l'$ after running the modified forward pass. The **excitatory interaction strength** is:

$$\mathcal{E}(f^{(l)}_i \to f^{(l')}_j) = \mathbb{E}_x\left[\frac{\partial \hat{f}^{(l')}_j}{\partial \alpha}\bigg|_{\alpha = f^{(l)}_i}\right]$$

If $\mathcal{E} > 0$, the interaction is excitatory. In practice, we approximate this with a finite difference:

$$\mathcal{E}(f^{(l)}_i \to f^{(l')}_j) \approx \mathbb{E}_x\left[\frac{\hat{f}^{(l')}_j(\alpha_{\text{high}}) - \hat{f}^{(l')}_j(\alpha_{\text{low}})}{\alpha_{\text{high}} - \alpha_{\text{low}}}\right]$$

where $\alpha_{\text{high}}$ and $\alpha_{\text{low}}$ are chosen relative to the feature's empirical activation distribution (e.g., the 90th percentile and 10th percentile of non-zero activations).

#### 3.2.2 Inhibition

Feature $f^{(l)}_i$ **inhibits** feature $f^{(l')}_j$ if $\mathcal{E}(f^{(l)}_i \to f^{(l')}_j) < 0$ — increasing $f^{(l)}_i$ causes $f^{(l')}_j$ to decrease.

Excitation and inhibition are the same measurement with opposite signs. We define the **signed interaction strength:**

$$I(f^{(l)}_i \to f^{(l')}_j) = \mathcal{E}(f^{(l)}_i \to f^{(l')}_j)$$

Positive $I$ = excitation. Negative $I$ = inhibition. The absolute value $|I|$ is the interaction magnitude.

#### 3.2.3 Gating (Modulatory Interaction)

Feature $f^{(l)}_i$ **gates** feature $f^{(l')}_j$ if $f^{(l)}_i$'s activation *modulates the sensitivity* of $f^{(l')}_j$ to its other inputs — i.e., the interaction is not purely additive but multiplicative.

This is more subtle and computationally interesting than pure excitation/inhibition. Formally, gating exists when:

$$\frac{\partial^2 \hat{f}^{(l')}_j}{\partial \alpha_i \, \partial \alpha_k} \neq 0$$

for some other feature $f^{(l'')}_k$ that also influences $f^{(l')}_j$. In other words, the effect of feature $k$ on feature $j$ *depends on* whether feature $i$ is active.

Operationally, we detect gating by measuring $f^{(l)}_i$'s excitatory/inhibitory effect on $f^{(l')}_j$ *conditioned on the activation of other features*:

$$G(f^{(l)}_i \text{ gates } f^{(l')}_j) = \text{Var}_{x}\left[I(f^{(l)}_i \to f^{(l')}_j \mid x)\right]$$

If the interaction strength $I$ has *high variance* across inputs (i.e., sometimes it's strongly excitatory, sometimes weakly, sometimes inhibitory), this suggests that the interaction is context-dependent — feature $i$ is gating feature $j$'s response to other inputs rather than having a fixed excitatory/inhibitory effect.

A more targeted test: partition inputs into those where a third candidate feature $f^{(l'')}_k$ is active vs. inactive, and compare the interaction strength:

$$G_k(f^{(l)}_i \text{ gates } f^{(l')}_j) = \left|I(f^{(l)}_i \to f^{(l')}_j \mid f^{(l'')}_k > 0) - I(f^{(l)}_i \to f^{(l')}_j \mid f^{(l'')}_k = 0)\right|$$

If this quantity is large, the three features participate in a gating interaction where $i$ and $k$ jointly determine $j$'s activation — a feature-level AND gate.

### 3.3 The Interaction Graph as a Formal Object

With these definitions, the **Feature Interaction Graph** is a weighted, directed, typed multigraph:

$$\mathcal{G} = (V, E_{\text{exc}}, E_{\text{inh}}, E_{\text{gate}})$$

where:
- $V = \{f^{(l)}_i : l \in \{1, \ldots, L\}, i \in \{1, \ldots, D_l\}\}$ is the set of all SAE features across all layers.
- $E_{\text{exc}} \subseteq V \times V \times \mathbb{R}^+$ is the set of excitatory edges with positive weights.
- $E_{\text{inh}} \subseteq V \times V \times \mathbb{R}^+$ is the set of inhibitory edges with positive weights (representing magnitude of inhibition).
- $E_{\text{gate}} \subseteq V \times V \times V \times \mathbb{R}^+$ is the set of gating edges (hyperedges involving three features) with weights.

Edges are directed: $f^{(l)}_i \to f^{(l')}_j$ only for $l' > l$ (or $l' = l$ for within-layer MLP-mediated interactions, depending on the SAE placement). This respects the causal ordering of the transformer's computation graph.

The graph is **sparse by construction**: we only include edges with $|I| > \tau$ for some threshold $\tau$, determined by a permutation test (shuffle feature activations to establish a null distribution of interaction strengths, and set $\tau$ at the 99th percentile of the null).

### 3.4 Relationship to the Underlying Transformer Computation

How do feature interactions relate to the actual attention and MLP operations?

The residual stream at layer $l'$ is:

$$\mathbf{h}^{(l')} = \mathbf{h}^{(l)} + \sum_{m=l+1}^{l'} \left(\text{Attn}^{(m)}(\mathbf{h}^{(m-1)}) + \text{MLP}^{(m)}(\mathbf{h}^{(m-1)} + \text{Attn}^{(m)}(\mathbf{h}^{(m-1)}))\right)$$

When we modify feature $f^{(l)}_i$ — adding $\Delta \alpha \cdot \mathbf{d}^{(l)}_i$ to $\mathbf{h}^{(l)}$ — this perturbation propagates through all subsequent attention and MLP operations. The resulting change in $f^{(l')}_j$ is mediated by:

1. **The residual stream path:** The perturbation $\Delta \alpha \cdot \mathbf{d}^{(l)}_i$ propagates directly through the residual stream to layer $l'$, where the SAE encoder projects it onto feature $j$'s direction. This gives a "direct connection" interaction proportional to $\mathbf{e}^{(l')}_j \cdot \mathbf{d}^{(l)}_i$ (the cosine alignment of the decoder direction at layer $l$ with the encoder direction at layer $l'$). For adjacent layers, this can be significant.

2. **Attention-mediated paths:** The perturbation alters the key/query/value computations in intermediate attention layers, potentially rerouting information flow. This is how feature interactions can be non-trivially complex — a feature that modifies attention patterns can redirect *other* information, creating indirect causal pathways.

3. **MLP-mediated paths:** The perturbation passes through nonlinear MLP layers (GeLU/SiLU activations), which means the effect can be amplified, suppressed, or gated depending on the operating point. This is the primary source of gating interactions: the nonlinearity of the MLP means that the effect of one perturbation depends on the current activation state, which is influenced by other features.

The key insight: **feature interactions are a compressed, interpretable summary of the full Jacobian of the transformer's forward pass**, projected into SAE feature space. Instead of a $d \times d$ Jacobian matrix at each layer transition (which is uninterpretable), we have a $D_l \times D_{l'}$ interaction matrix in feature space (which is interpretable because each row and column corresponds to a named, understood concept).

---

## 4. The Combinatorial Challenge and How to Solve It

### 4.1 The Scale of the Problem

A typical SAE dictionary has $D \sim 16,\!384$ to $131,\!072$ features per layer. A model with $L = 26$ layers (Gemma-2-2B) has $L \times D \sim 400,\!000$ to $3.4$ million total features. Testing all pairwise interactions across all layer pairs is $O(L^2 D^2)$ — approximately $10^{11}$ to $10^{13}$ pairs. At $\sim$100ms per intervention (forward pass + modification), this would take $\sim$300,000 GPU-years. Obviously intractable.

This is the central technical challenge, and it's why no one has built the interaction graph. But it is solvable. The key observation: **real feature interaction graphs are sparse.** Not every feature interacts with every other feature. Most features at layer 5 have negligible causal effect on most features at layer 20. The problem is identifying *which* pairs interact without testing all of them.

### 4.2 Pruning Strategy 1: Co-Activation Filtering

**Principle:** Features that never co-activate on the same inputs cannot have observationally relevant interactions.

**Method:** Run a large corpus ($\sim$10M tokens) through the model and all SAEs. For each pair of features $(f^{(l)}_i, f^{(l')}_j)$, compute the conditional probability:

$$P(f^{(l')}_j > 0 \mid f^{(l)}_i > 0)$$

Only test interactions where this co-activation probability exceeds a threshold (e.g., $> 0.01$). This eliminates pairs that are topically unrelated — a "DNA sequence" feature and a "French cuisine" feature are unlikely to interact because they rarely co-occur.

**Expected pruning:** Empirically, SAE features are sparse — each feature fires on a small fraction of inputs ($\sim$0.1–5%). Co-activation is even sparser. This typically prunes $>$99% of candidate pairs.

**Caveat:** This misses *inhibitory* interactions where one feature suppresses the other (they would have *lower* than baseline co-activation, and the co-activation filter would discard them). To catch inhibitory interactions, also flag pairs with anomalously *low* co-activation: $P(f^{(l')}_j > 0 \mid f^{(l)}_i > 0) \ll P(f^{(l')}_j > 0)$, which suggests feature $i$ may be suppressing feature $j$.

### 4.3 Pruning Strategy 2: Decoder Direction Alignment

**Principle:** A feature at layer $l$ can only directly influence a feature at layer $l'$ if the decoder direction at layer $l$ has non-negligible projection onto the encoder direction at layer $l'$ (after accounting for intermediate computations).

**Method:** For adjacent layers, compute:

$$\text{DirectAlign}(f^{(l)}_i, f^{(l+1)}_j) = |\mathbf{e}^{(l+1)}_j \cdot \mathbf{d}^{(l)}_i|$$

where $\mathbf{e}^{(l+1)}_j$ is the encoder direction for feature $j$ at layer $l+1$ and $\mathbf{d}^{(l)}_i$ is the decoder direction for feature $i$ at layer $l$. Features with high alignment have a "direct wire" through the residual stream.

For non-adjacent layers, this becomes less informative because intermediate computations can rotate, amplify, or suppress directions. But the direct alignment still serves as a useful prior — pairs with zero direct alignment are less likely to have strong interactions (though attention-mediated paths can create interactions between features with orthogonal decoder/encoder directions).

**Expected pruning:** Decoder directions in high-dimensional space ($d = 2048$ for Gemma-2-2B) are mostly orthogonal. Only a small fraction of cross-layer pairs will have significant alignment. Combined with co-activation filtering, this can reduce the candidate set by another order of magnitude.

### 4.4 Pruning Strategy 3: Importance-Weighted Sampling

**Principle:** Not all features are equally important. Focus first on features that matter most for model behavior.

**Method:** Rank features by a composite importance score:

$$\text{Importance}(f^{(l)}_i) = \text{ActivationFreq}(f^{(l)}_i) \times \text{DownstreamCausalEffect}(f^{(l)}_i)$$

where:
- $\text{ActivationFreq}$ is the fraction of tokens on which the feature is non-zero.
- $\text{DownstreamCausalEffect}$ is the mean absolute change in the model's output (logit distribution or loss) when the feature is ablated.

Select the top-$K$ features (e.g., $K = 200$) by importance score. Test interactions only among these top features. This gives $K^2 = 40,\!000$ candidate pairs — tractable even with 100ms per intervention.

**Justification:** The heavy-tailed distribution of feature importance means that a small fraction of features account for a large fraction of the model's computation. The interaction graph among these high-importance features captures the "backbone" of the model's computational architecture; lower-importance features can be filled in later.

### 4.5 Pruning Strategy 4: Layer Locality

**Principle:** Most causal interactions are local — features at layer $l$ primarily influence features at layers $l+1$ through $l+k$ for small $k$, with influence decaying with distance.

**Method:** Only test interactions within a layer window: $|l' - l| \leq W$ for some window size $W$ (e.g., $W = 5$). This reduces the number of layer pairs from $O(L^2)$ to $O(L \times W)$.

**Justification:** While the residual stream theoretically propagates information across all layers, attention patterns and MLP operations at each layer primarily process information from the recent residual stream. Long-range interactions (feature at layer 2 directly influencing feature at layer 24) almost certainly exist but are likely mediated by chains of shorter-range interactions — which the local graph will still capture as multi-hop paths.

### 4.6 Combining All Strategies

The full pruning pipeline:

1. **Importance filter:** Select top-$K$ features ($K = 200$–$500$). $\to K^2 \times L \times W \sim 10^6$–$10^7$ candidate pairs.
2. **Co-activation filter:** Remove pairs with negligible co-activation. $\to$ Typical 90-99% reduction $\to 10^4$–$10^6$ pairs.
3. **Decoder alignment filter:** Remove pairs with near-zero direct alignment (for adjacent layers). $\to$ Further 50-90% reduction $\to 10^3$–$10^5$ pairs.
4. **Run interventions** on remaining candidates. At 100ms each, $10^4$ pairs $\times$ 100 examples each = $10^6$ forward passes $\approx$ 28 GPU-hours on an A100. **Feasible.**

For a model like GPT-2-small (12 layers, 768-dimensional residual stream, existing public SAEs with $D = 24,\!576$ features per layer), the full pipeline is comfortably runnable on a single consumer GPU within a week.

---

## 5. The Computational Mechanisms Behind Feature Interactions

Understanding *why* feature interactions arise in transformers is essential for interpreting the graph structure. There are four distinct mechanisms through which feature $f^{(l)}_i$ can influence feature $f^{(l')}_j$.

### 5.1 Mechanism 1: Residual Stream Propagation (Direct)

The simplest mechanism. The decoder direction $\mathbf{d}^{(l)}_i$ is added to the residual stream and propagates forward. At layer $l'$, the SAE encoder projects the residual stream onto feature $j$'s direction. If $\mathbf{d}^{(l)}_i$ has nonzero projection onto $\mathbf{e}^{(l')}_j$ (after accounting for LayerNorm), the interaction is direct.

**Characteristics:**
- Interaction strength is approximately linear in the source feature's activation.
- Interaction exists even with no intermediate attention or MLP computation (pure residual propagation).
- Captured by the decoder-alignment pruning filter.
- Likely dominates for *adjacent* layers and for features in *similar* semantic domains (whose directions are aligned).

**Mathematical form:**

$$I_{\text{direct}}(f^{(l)}_i \to f^{(l+1)}_j) \approx \frac{\partial f^{(l+1)}_j}{\partial f^{(l)}_i} = \mathbf{e}^{(l+1)}_j \cdot \text{LN}'^{(l+1)} \cdot \mathbf{d}^{(l)}_i$$

where $\text{LN}'^{(l+1)}$ is the Jacobian of the LayerNorm at layer $l+1$ (which depends on the input, making even "direct" interactions slightly input-dependent).

### 5.2 Mechanism 2: Attention-Mediated Interaction

A richer mechanism. Modifying feature $f^{(l)}_i$ at one token position changes the key, query, or value representations computed by attention heads in layers $l+1$ through $l'$. This can:

- **Reroute attention:** If $\mathbf{d}^{(l)}_i$ has significant projection onto a key or query direction of some attention head, modifying the feature changes the attention pattern. This redirects information flow from *other* token positions, potentially activating or suppressing features at downstream layers that depend on information from those positions.
- **Modify values:** If $\mathbf{d}^{(l)}_i$ projects onto a value direction, the feature modification directly changes what information the attention head writes into the residual stream at attended-to positions.

**Characteristics:**
- Can create long-range interactions (feature at one position affects feature at a distant position, mediated by attention).
- Interaction strength depends on the attention pattern, which depends on the input. This is a source of context-dependent (gating) interactions.
- Can create interactions between features with *orthogonal* decoder/encoder directions, as long as an intermediate attention head serves as a "translator."

**This is likely the primary mechanism for cross-token feature interactions** — e.g., a "proper noun" feature at position 5 exciting a "factual recall" feature at position 12, mediated by an attention head that routes information from names to predicates.

### 5.3 Mechanism 3: MLP-Mediated Interaction (Nonlinear)

MLP layers apply a nonlinear transformation (typically GeLU or SiLU): $\text{MLP}(\mathbf{x}) = W_{\text{out}} \cdot \sigma(W_{\text{in}} \cdot \mathbf{x} + \mathbf{b})$. The nonlinearity means that the effect of modifying one feature depends on the *current activation state* of the MLP neurons, which in turn depends on other features.

This is the mechanistic origin of **gating interactions**: feature $i$ modifies the residual stream, pushing some MLP neurons into or out of their active regime, which changes how those neurons respond to *other* features, thereby modulating feature $j$'s activation.

**Characteristics:**
- Inherently nonlinear — cannot be captured by linear approximations alone.
- Produces context-dependent interaction strengths (gating).
- Can implement AND-gate-like computations: feature $j$ fires only when *both* feature $i$ and feature $k$ are active, because the MLP neuron that writes into $j$'s direction requires both components of its input to exceed the GeLU threshold.

**Mathematical sketch:** Consider a single MLP neuron with pre-activation $z = \mathbf{w}_{\text{in}}^T \mathbf{h}^{(l)} + b$. If feature $f^{(l)}_i$ contributes a component $f^{(l)}_i (\mathbf{w}_{\text{in}}^T \mathbf{d}^{(l)}_i)$ to $z$, then the effect of changing $f^{(l)}_i$ on the neuron's output depends on $\sigma'(z)$ — the derivative of the activation function at the *current operating point*. If $z$ is near the GeLU's transition region, a small change in $f^{(l)}_i$ can switch the neuron on or off, dramatically affecting downstream features.

### 5.4 Mechanism 4: LayerNorm-Mediated Interaction (Global)

LayerNorm normalizes the residual stream to have zero mean and unit variance at each layer. This creates a subtle global interaction: *increasing any feature's activation slightly suppresses all other features' contributions to the normalized stream*, because the normalization denominator grows.

$$\text{LN}(\mathbf{h}) = \frac{\mathbf{h} - \mu(\mathbf{h})}{\sigma(\mathbf{h})} \cdot \gamma + \beta$$

When feature $i$'s decoder direction $\mathbf{d}^{(l)}_i$ is added to the residual stream, it changes $\mu$ and $\sigma$, which modifies the normalized representation of *every* feature's contribution. This creates a weak, global inhibitory interaction — every feature mildly inhibits every other feature via the normalization bottleneck.

**Characteristics:**
- Universally present but typically weak (the normalization denominator changes by a small fraction when one feature changes).
- Creates a "budget" effect: increasing one feature's activation effectively reduces the fractional contribution of all others.
- Likely not the dominant interaction mechanism for any specific pair, but important for understanding global graph properties (e.g., an overall inhibitory "background" that makes the effective excitatory interactions even more significant).

---

## 6. Research Program: Five Interlocking Experiments

### 6.1 Experiment 1: Co-Activation Atlas — The Correlational Scaffold

**Goal:** Build a comprehensive co-activation map as the "null model" against which causal interactions are measured, and as the primary filter for identifying candidate interaction pairs.

**Method:**

1. Select a target model with high-quality pre-trained SAEs: **Gemma-2-2B with Gemma Scope SAEs** (released by DeepMind, 16K and 65K feature dictionaries at every layer) or **GPT-2-small with community SAEs** (e.g., from Joseph Bloom's SAELens or EleutherAI).
2. Run a diverse corpus (~10M tokens from a mix of Wikipedia, code, dialogue, and creative writing) through the model and record SAE feature activations at every layer for every token position.
3. For each pair of features $(f^{(l)}_i, f^{(l')}_j)$ with $|l' - l| \leq 5$:
   - Compute co-activation frequency: $P(f^{(l')}_j > 0 \mid f^{(l)}_i > 0)$.
   - Compute conditional mean activation: $\mathbb{E}[f^{(l')}_j \mid f^{(l)}_i > 0]$ vs. $\mathbb{E}[f^{(l')}_j]$.
   - Compute pointwise mutual information: $\text{PMI}(f^{(l)}_i, f^{(l')}_j) = \log \frac{P(f^{(l)}_i > 0, f^{(l')}_j > 0)}{P(f^{(l)}_i > 0) \cdot P(f^{(l')}_j > 0)}$.
4. **Cluster features by co-activation profile:** Use the co-activation matrix to identify "feature communities" — groups of features that consistently co-activate. These are candidate "computational modules."
5. **Identify anomalously low co-activation pairs:** Pairs where $P(f^{(l')}_j > 0 \mid f^{(l)}_i > 0) \ll P(f^{(l')}_j > 0)$, suggesting mutual inhibition.

**Output:** A co-activation atlas: a sparse matrix of PMI values for all feature pairs within the layer window, plus a community detection analysis. This is *not* the interaction graph (correlation ≠ causation), but it provides the scaffold on which the causal graph is built.

**Compute:** Feature activation collection is a single forward pass per example — ~10M tokens at ~100 tokens/sec on an A100 = ~28 hours. Co-activation statistics are computed offline (sparse matrix operations). Very feasible.

### 6.2 Experiment 2: Causal Interaction Measurement via Feature Clamping

**Goal:** Measure the *causal* interaction strength between candidate feature pairs identified in Experiment 1, producing the core interaction graph.

**Method:**

1. From the co-activation atlas, select the top-$N$ candidate pairs ($N \sim 5,\!000$–$50,\!000$) based on high PMI, anomalously low co-activation, or high decoder-direction alignment.
2. Also include the top-$K$ most important features ($K \sim 200$) and test all pairs among them.
3. For each candidate pair $(f^{(l)}_i, f^{(l')}_j)$, run the feature clamping intervention:
   - Sample $M$ inputs ($M \sim 100$) where $f^{(l)}_i$ is naturally active (activation > 50th percentile of its non-zero activations).
   - For each input, run two forward passes:
     - **Baseline:** normal forward pass, record $f^{(l')}_j$'s activation.
     - **Intervention:** clamp $f^{(l)}_i$ to its mean non-zero activation (or to zero, for the inhibition test), record $f^{(l')}_j$'s activation.
   - Compute the mean difference: $I(f^{(l)}_i \to f^{(l')}_j) = \mathbb{E}_x[\hat{f}^{(l')}_j(\text{clamped}) - f^{(l')}_j(\text{baseline})]$.
   - Compute the variance of this difference across inputs (for gating detection).
4. **Statistical testing:** Use a permutation test (shuffle clamping assignments) to establish a null distribution of interaction strengths. Retain only edges with $p < 0.01$ after Bonferroni correction for multiple comparisons.
5. **Type classification:** For each significant edge:
   - $I > 0$: excitatory (signed weight = $I$).
   - $I < 0$: inhibitory (signed weight = $I$).
   - High variance of $I$ across inputs (relative to mean): gating interaction. Further investigate by conditioning on candidate third features.

**Output:** The Feature Interaction Graph $\mathcal{G}$: a sparse, directed, weighted graph with typed edges. Each edge has a measured causal strength, a p-value, an interaction type (excitatory/inhibitory/gating), and associated metadata (the layer pair, the feature identities and descriptions, examples where the interaction is strongest).

**Compute:** $50,\!000$ pairs $\times$ $100$ inputs $\times$ $2$ forward passes = $10^7$ forward passes. On GPT-2-small ($\sim$0.2ms per forward pass on A100 with batch processing), this is $\sim$0.5 GPU-hours. On Gemma-2-2B ($\sim$5ms per forward pass), this is $\sim$14 GPU-hours. **Very feasible.**

### 6.3 Experiment 3: Graph-Theoretic Analysis of the Interaction Network

**Goal:** Characterize the global structure of the feature interaction graph using tools from network science. Determine whether the graph has the properties expected of a functional computational architecture.

**Method:**

1. Compute standard graph statistics:
   - **Degree distribution:** Is it heavy-tailed (scale-free)? If so, there are "hub features" that participate in many interactions — critical nodes in the computational architecture.
   - **Clustering coefficient:** Do features form tightly interconnected clusters (computational modules)?
   - **Shortest path distribution:** What is the typical number of hops between two features? A small diameter ("small-world" property) would suggest efficient information propagation.
   - **Assortativity:** Do high-degree features preferentially connect to other high-degree features (hierarchical structure) or to low-degree features (hub-and-spoke structure)?
   - **Reciprocity:** Among features at the same layer distance, how often is the interaction bidirectional ($i \to j$ and $j \to i$)? High reciprocity suggests feedback-like computation; low reciprocity suggests feedforward processing.

2. **Community detection:** Apply modularity-based community detection (Louvain or Leiden algorithm) to identify groups of features that interact more strongly within the group than between groups. These communities are candidate "functional modules" — groups of features that collectively implement a coherent computation.

3. **Excitatory vs. inhibitory subgraph analysis:** Separate the graph into its excitatory and inhibitory subgraphs. Are they structurally different? In biological neural networks, excitatory and inhibitory connections have different topological properties (inhibitory connections are more local and nonspecific, excitatory connections are more long-range and specific). Does the same hold for feature interactions?

4. **Layer-distance analysis:** How does interaction strength decay with layer distance? Plot mean $|I|$ as a function of $|l' - l|$. Exponential decay would suggest local computation; power-law decay would suggest long-range "skip connection"-like interactions through the residual stream.

5. **Hub identification:** Identify features with high in-degree (many features influence them) and high out-degree (they influence many features). These hub features are the "critical infrastructure" of the model's computation — understanding them is high-leverage.

**Expected findings (hypotheses to test):**

- **Scale-free or heavy-tailed degree distribution:** Expected by analogy with other evolved/optimized networks. A few hub features participate in many computations; most features have limited interaction neighborhoods.
- **Modular structure with inter-module hubs:** Features cluster into semantic/functional modules (e.g., a "language style" module, a "factual knowledge" module, a "reasoning" module) connected by hub features that integrate information across modules.
- **Asymmetric excitation/inhibition topology:** Excitatory interactions are more specific and long-range (connecting features that are semantically related across layers); inhibitory interactions are more local and nonspecific (implementing competition within a layer's feature space).

### 6.4 Experiment 4: Behavior-Specific Subgraph Extraction

**Goal:** For known model behaviors, extract the relevant subgraph from the interaction graph and validate that it constitutes a coherent "feature-level circuit."

**Method:**

1. Select 3–4 well-characterized model behaviors:
   - **Factual recall:** "The capital of France is [Paris]" — requires retrieving a specific fact.
   - **Refusal:** "How do I make a [harmful request]?" → "I can't help with that" — requires detecting harm and generating refusal.
   - **Induction / in-context learning:** "[A B ... A] → B" — requires pattern matching and copying.
   - **Sentiment-influenced generation:** "This movie was terrible. The acting was [awful]" — requires propagating sentiment to word choice.

2. For each behavior, identify the "seed features" — features that are strongly activated on inputs exhibiting that behavior and causally important for the behavior (via activation patching on the model output).

3. Extract the **behavior subgraph:** starting from the seed features, follow all edges (excitatory and inhibitory) with strength above a threshold, recursively expanding to connected features up to $k$ hops. This gives the "feature-level circuit" for that behavior.

4. **Validate the subgraph:**
   - **Sufficiency test:** If we clamp all features *outside* the subgraph to their mean activations (effectively ablating them), does the behavior persist? If yes, the subgraph is (approximately) sufficient.
   - **Necessity test:** If we ablate features *inside* the subgraph (clamp to zero), does the behavior degrade? If yes, the subgraph is necessary.
   - **Interpretability test:** Does the causal flow through the subgraph make semantic sense? E.g., for factual recall, do we see: "entity mention" feature → (excites) "entity attribute" feature → (excites) "specific fact" feature → (excites) "output token" feature?

5. **Compare to head-level circuits:** For behaviors where head-level circuits have been published (IOI, induction), compare the feature-level circuit to the known head-level circuit. Are the same components identified? Does the feature-level circuit reveal finer-grained structure within heads that the head-level analysis missed?

**Expected output:** Annotated feature-level circuit diagrams for 3–4 behaviors, with validated necessity/sufficiency and side-by-side comparison to head-level circuits. This is the first demonstration that the interaction graph provides a *useful decomposition* of model computation — turning it from an abstract graph into a practical interpretability tool.

### 6.5 Experiment 5: Interaction Graph-Guided Steering

**Goal:** Use the interaction graph to predict and control the cascading effects of feature-level interventions, demonstrating practical utility.

**Method:**

1. Select a steering target: e.g., "make the model more formal" (by amplifying a "formal register" feature).

2. **Naive steering (baseline):** Clamp the target feature to a high activation. Measure the effect on model output (perplexity, behavioral change, unintended side effects on other behaviors).

3. **Graph-informed steering:** Using the interaction graph, predict the cascading effects of the target intervention:
   - Which features will be excited? (Follow excitatory edges from the target.)
   - Which features will be inhibited? (Follow inhibitory edges.)
   - Are any undesired features in the cascade? (E.g., does "formal register" excite "verbose" or inhibit "helpful"?)

4. **Compensated steering:** Identify undesired cascading effects from Step 3 and apply compensatory interventions — simultaneously steer the target feature up while clamping the undesired downstream features to their baseline values. Measure whether this produces cleaner, more targeted behavioral change than naive steering.

5. **Quantitative comparison:**
   - **Steering precision:** On a held-out evaluation set, measure target behavior change per unit of unintended side effect.
   - **Prediction accuracy:** How well does the interaction graph predict the actual downstream feature changes from the intervention? Compute correlation between predicted and observed feature activation changes.

**Expected output:** Demonstration that the interaction graph enables more precise model steering by predicting and compensating for cascading effects. This is the "killer app" that demonstrates the practical value of the interaction graph beyond scientific understanding.

---

## 7. Why This Hasn't Been Done: The Real Barriers

### 7.1 The Two-Community Gap

SAE research and circuit analysis are practiced by overlapping but distinct subcommunities:

- **SAE researchers** focus on dictionary learning, feature quality, reconstruction loss, and individual feature interpretability. Their unit of analysis is the *individual feature*. Their tools are SAELens, Neuronpedia, and activation analysis pipelines.
- **Circuit researchers** focus on attention heads, MLPs, and information flow using activation patching, path patching, and causal scrubbing. Their unit of analysis is the *head/MLP component*. Their tools are TransformerLens, Baukit, and patchscope-type interventions.

The interaction graph sits *exactly between* these two communities: it uses SAE features (from the SAE community) as nodes but measures causal interactions (from the circuit community) as edges. Neither community has naturally built the bridge.

Anthropic's attribution graphs are the closest attempt, but they were done *internally* with custom infrastructure that isn't fully open-sourced, and they focus on per-prompt tracing rather than structural graph construction.

### 7.2 The Combinatorial Fear

The $O(D^2 L^2)$ scaling of naive pairwise interaction testing has discouraged attempts. Researchers see "millions of features" and conclude the project is intractable. As shown in Section 4, this fear is solvable with smart pruning — but the solution requires combining techniques from network science (community detection), statistics (multiple testing correction), and ML engineering (batched interventions), which isn't standard toolkit for most MI researchers.

### 7.3 No Existing Formalism

Until this project, there has been no published formal definition of "feature interaction" at the SAE level. The formalism in Section 3 — distinguishing excitation, inhibition, and gating; defining interaction strength via clamping interventions; specifying the graph as a typed multigraph — is new. Without this formalism, it's hard to even *propose* the project concretely enough to execute.

### 7.4 The "Single Feature is Hard Enough" Mindset

Many researchers feel that we haven't yet fully understood *individual* features (polysemanticity, feature splitting, absorption, composition at the single-feature level) and that jumping to interactions is premature. This is a reasonable concern but ultimately misguided — it's the same argument as "we haven't finished cataloging genes, so studying gene regulation is premature." In practice, understanding interactions illuminates individual features (a feature's role only becomes clear in the context of its interaction neighborhood), and waiting for individual-feature understanding to be "complete" means waiting forever.

---

## 8. Connection to the Broader MI Research Agenda

### 8.1 Feature Interaction Graphs and Reward Model Interpretability

The reward model interpretability project (see companion document) asks "what writes into the reward direction?" The interaction graph answers a superset of this question: it maps *all* structural relationships between *all* features, of which the reward-relevant subgraph is a special case. For a reward model, the interaction graph would reveal the complete "preference computation architecture" — which feature communities contribute to reward, how they compose, and which interactions create hackable shortcuts.

More specifically: the reward lens gives us per-feature reward attribution ($\mathbf{w}_r^T \mathbf{d}_i$), but it doesn't tell us *why* feature $i$ has high reward attribution. The interaction graph answers this: feature $i$ has high reward attribution because it is excited by features $j$ and $k$ (which detect content quality and safety), and it inhibits feature $m$ (which detects harm). The interaction graph gives the *causal explanation* behind the reward attribution.

### 8.2 Feature Interaction Graphs and Circuit Universality

The Rosetta Circuits project asks whether circuits are universal across model families. The interaction graph provides a natural tool for this comparison: instead of comparing model-specific attention head labels, we compare *feature interaction graphs*. If two models have interaction subgraphs for "factual recall" with the same topology (same node types, same edge types, same information flow pattern) despite different architectures, that's strong evidence for universality at the feature level — arguably stronger evidence than comparing head-level circuits, because features are more semantically interpretable and architecture-independent than heads.

### 8.3 Feature Interaction Graphs and CoT Faithfulness

The CoT Faithfulness Dissector asks whether the "answer computation" and the "CoT generation" use the same circuits. In feature interaction graph terms, this becomes: does the subgraph activated during answer computation overlap with the subgraph activated during CoT generation? A structural interaction graph enables a cleaner version of this analysis — instead of comparing activation-patching effects on different outputs, we compare the *structural neighborhood* of "answer-relevant" features with the structural neighborhood of "CoT-generation" features.

### 8.4 Feature Interaction Graphs as Foundational Infrastructure

Ultimately, the interaction graph is not just a research project — it is **infrastructure**. Just as Neuronpedia provides a browsable catalog of individual features, the interaction graph provides a browsable map of *relationships between features*. Every downstream MI project benefits from knowing which features are structurally connected, just as every molecular biology project benefits from knowing which proteins interact.

---

## 9. What the Interaction Graph Would Look Like (Concrete Predictions)

Before executing the experiments, it's worth stating concrete, falsifiable predictions about the interaction graph's structure. This serves as a pre-registration of sorts and ensures the project produces clear positive or negative results regardless of outcome.

### 9.1 Prediction: The Graph is Sparse

**Claim:** Fewer than 1% of tested feature pairs will have statistically significant causal interactions. The interaction graph is not a "dense mesh" but a sparse network with clear structure.

**Rationale:** The residual stream is high-dimensional ($d = 768$ for GPT-2-small, $d = 2048$ for Gemma-2-2B), and SAE features correspond to nearly orthogonal directions. A perturbation along one feature direction propagates through attention and MLPs, but the projection onto most other feature directions is negligible. Only pairs with significant alignment (direct or via intermediate computations) will produce measurable interactions.

**If wrong:** If the graph is dense (most pairs interact significantly), it would suggest that features are not the right "atomic units" for understanding computation — the real units might be feature *combinations* or subspaces, and individual features are too fine-grained.

### 9.2 Prediction: Hub Features Exist and Are Functionally Meaningful

**Claim:** The degree distribution is heavy-tailed, with a small number of "hub features" (high in-degree + out-degree) that participate in many interactions. These hubs correspond to semantically general features (e.g., "sentence boundary," "entity mention," "negation") rather than semantically specific ones ("Golden Gate Bridge," "recursive Python function").

**Rationale:** General-purpose features are used across many computations and should therefore have many interaction edges. Specific features are used in narrow contexts and should have few edges. This mirrors the hub structure in biological networks, where housekeeping genes have high connectivity and tissue-specific genes have low connectivity.

**If wrong:** If there are no hubs (uniform degree distribution), it would suggest a surprisingly "democratic" computational architecture with no bottleneck features — which would be good news for steering (no single point of failure) but would challenge some models of how transformers organize computation.

### 9.3 Prediction: Excitatory Interactions Dominate Across Layers, Inhibitory Interactions Dominate Within Layers

**Claim:** Cross-layer interactions (layers $l \to l+k$, $k > 0$) are predominantly excitatory — features at earlier layers "activate" features at later layers. Within-layer interactions (same layer, mediated by the MLP) are more balanced between excitation and inhibition, reflecting competition among features for representation in the residual stream.

**Rationale:** The feedforward computation of a transformer builds representations incrementally — each layer adds information. This naturally produces excitatory cross-layer interactions (feature detection builds on prior feature detection). Within a layer, however, features compete for the limited capacity of the residual stream (superposition pressure), creating inhibitory dynamics.

**If wrong:** If cross-layer interactions are predominantly inhibitory, it would suggest a "gating cascade" architecture where each layer selectively suppresses prior representations rather than building on them — a very different computational model than the standard "residual stream as information accumulator" picture.

### 9.4 Prediction: Behavior-Specific Subgraphs are Modular

**Claim:** The feature-level circuit for a specific behavior (e.g., factual recall) is a well-defined subgraph with clear boundaries — the features inside the subgraph are densely connected to each other and sparsely connected to features outside it.

**Rationale:** Modularity is a prediction of the superposition hypothesis — models learn approximately modular circuits that share features under capacity pressure but maintain functional separability. If circuits are modular at the feature level, it supports the view that complex behaviors decompose into relatively independent computational modules.

**If wrong:** If behavior subgraphs are not modular (every behavior involves a diffuse, overlapping soup of features with no clear boundaries), it would suggest that the "circuit" abstraction breaks down at the feature level, and computation is fundamentally distributed in a way that resists decomposition. This would be a deeply important negative result for the field.

### 9.5 Prediction: The Interaction Graph Predicts Steering Side Effects

**Claim:** When a feature is steered (clamped to a non-baseline activation), the features that change most in response are the ones with the strongest edges *from* the steered feature in the interaction graph. The interaction graph predicts at least 50% of the variance in downstream feature activation changes.

**Rationale:** If the interaction graph captures the model's computational dependencies, then perturbations should propagate along the graph's edges. This is essentially a test of whether the graph is a good model of the model's causal structure.

**If wrong:** If the interaction graph fails to predict steering cascades, it suggests that the linear clamping interventions used to build the graph don't capture the full nonlinear dynamics of the model, and more sophisticated interaction measures are needed (e.g., second-order effects, path-specific interventions).

---

## 10. Compute Requirements and Feasibility

| Component | GPT-2-small | Gemma-2-2B | Hardware |
|-----------|-------------|------------|----------|
| Co-activation atlas (10M tokens) | ~4 hours | ~28 hours | 1× A100 (inference) |
| Candidate pair identification | ~1 hour (CPU) | ~4 hours (CPU) | CPU only |
| Causal interaction measurement (50K pairs × 100 inputs) | ~0.5 hours | ~14 hours | 1× A100 |
| Graph analysis | ~1 hour (CPU) | ~2 hours (CPU) | CPU only |
| Behavior subgraph extraction + validation | ~2 hours | ~8 hours | 1× A100 |
| Steering experiments | ~1 hour | ~4 hours | 1× A100 |
| **Total** | **~10 hours** | **~56 hours** | **1× A100** |

**Fully feasible on a single A100 GPU within one week for GPT-2-small, two weeks for Gemma-2-2B.** No training is required — all experiments use pre-trained models and pre-trained SAEs. The GPT-2-small version can even run on a consumer GPU (RTX 3090/4090) with batching optimizations, since inference on a 124M-parameter model is extremely cheap.

---

## 11. Deliverables

1. **Paper: "The Feature Interaction Graph: Mapping Compositional Computation in SAE Feature Space"** — The first systematic construction and analysis of a structural (prompt-independent) feature interaction graph, including graph statistics, behavior-specific subgraph extraction, and steering applications.

2. **Open-source library: `feature-graph`** — A toolkit extending SAELens / TransformerLens with:
   - Batched co-activation computation and atlas construction.
   - Feature clamping intervention infrastructure with statistical testing.
   - Interaction graph construction, filtering, and export (NetworkX, graph-tool compatible).
   - Subgraph extraction for user-specified behaviors.
   - Steering cascade prediction and compensated steering utilities.

3. **Interactive visualization: Feature Interaction Explorer** — A web-based tool (think Neuronpedia but for *relationships*) where users can:
   - Browse the interaction graph for a given model.
   - Click on a feature and see its excitatory/inhibitory/gating neighbors.
   - Select a behavior and see the extracted subgraph.
   - Simulate a steering intervention and visualize predicted cascades.
   - Compare interaction neighborhoods across models (if built for multiple models).

4. **Pre-computed interaction graphs** for GPT-2-small and Gemma-2-2B, released as downloadable datasets. These become community resources — other researchers can use them to study specific phenomena without re-running the full pipeline.

5. **A "Feature Grammar" taxonomy** — A categorization of recurring interaction motifs (e.g., "excitatory chain," "inhibitory competition," "gating triangle," "feedback loop") with examples from the interaction graph, establishing a vocabulary for describing compositional computation in feature space.

---

## 12. Why Now

- **Pre-trained SAE dictionaries are newly available at quality and scale.** Gemma Scope (DeepMind, 2024) provides SAEs at every layer of Gemma-2-2B and Gemma-2-9B. Community SAEs for GPT-2, Pythia, and Llama are mature. For the first time, we can study feature interactions *without first having to train SAEs* — a months-long prerequisite that previously blocked this research.

- **The tooling ecosystem has matured.** SAELens provides standardized SAE training and inference. TransformerLens provides intervention infrastructure. Baukit and nnsight provide alternative intervention frameworks. The engineering barrier to running thousands of clamping interventions has dropped from "build everything from scratch" to "write a loop."

- **Anthropic's attribution graphs have primed the community.** The "Biology of a Large Language Model" publication demonstrated that tracing computation through SAE features is possible and produces interpretable results. This created demand for the next step — going from per-prompt traces to structural maps — which is exactly this project.

- **The space is nearly empty.** One workshop paper (Manning-Coe et al., NeurIPS MechInterp Workshop 2025) on feature interactions in crosscoders. That's it. The first comprehensive, structural feature interaction graph will define this subfield.

- **The safety applications are immediate.** As SAE-based steering becomes more common in practice (see Anthropic's feature-based model control work), understanding cascading effects becomes a practical safety requirement, not just a scientific curiosity. The interaction graph is the tool that makes steering predictable.

The parts catalog is built. The time has come to draw the wiring diagram.
