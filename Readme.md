# Final Year Project — Memory-Augmented GPT-2: Work Completed, Diagnosis, and Current Plan

## 1. What are we trying to build?

The project is a **Memory-Augmented GPT-2** in which GPT-2 has access to a small external memory bank.

The basic objective is:

> Can a language model selectively store useful information in a bounded external memory, maintain diverse memory slots, and later retrieve the correct stored information when it becomes relevant?

Our architecture contains:

\[
\boxed{
\text{GPT-2}
+\text{Writer}
+\text{Router}
+\text{Memory Bank}
+\text{Reader}
+\text{Fusion}
}
\]

The memory currently contains:

\[
N=8\text{ slots},\qquad d=768
\]

so:

\[
M\in\mathbb{R}^{8\times768}.
\]

The overall process is:

```text
Input
  ↓
GPT-2 hidden states
  ↓
Summary representation
  ↓
Router ───────────────┐
  ↓                   │
Candidate Writer      │
  ↓                   │
Write Gate            │
  ↓                   │
Orthogonal Update     │
  ↓                   │
Memory Bank ←─────────┘
  ↓
Reader
  ↓
Memory context
  ↓
Fuse with GPT-2 hidden state
  ↓
LM Head
  ↓
Next-token prediction
```

---

# 2. Initial language-model experiments

Before concentrating on retrieval, we tested whether adding external memory helped normal language modelling.

Our controlled results included approximately:

| Model | Validation PPL ↓ |
|---|---:|
| Plain GPT-2 | ~33.15 |
| Scalar-gated memory | ~31.84 |
| Vector-gated memory | ~29.60 |

So the memory-augmented model gave better perplexity than the plain GPT-2 baseline.

This told us:

> **The memory architecture can affect the language model beneficially.**

But it did **not** prove that the model was actually storing a particular fact and retrieving that fact later.

That distinction became very important.

---

# 3. Scalar gate vs Vector gate

One of the main original research questions was whether a scalar gate would cause homogeneous memory behaviour compared with a slot-wise vector gate.

Later controlled results were:

| Metric | Scalar | Vector |
|---|---:|---:|
| PPL ↓ | **29.6346** | 29.6390 |
| Effective Rank ↑ | 5.932 | **7.206** |
| Stable Rank ↑ | 1.470 | **2.626** |
| Pairwise Cosine ↓ | 0.691 | **0.305** |
| Gate Slot Variance | 0.000 | **0.0343** |

### Interpretation

Perplexity was almost identical.

But the internal memory geometry was substantially better with vector gating.

Scalar gating produced more similar slots:

\[
\cos(M_i,M_j)\approx0.691
\]

while vector gating reduced this to:

\[
\approx0.305.
\]

Effective and stable rank also increased.

Therefore:

> **Vector gating improved memory diversity, even though this did not translate into a meaningful PPL difference.**

This was one of our first important findings:

\[
\boxed{\text{Better memory geometry}\neq\text{better retrieval automatically}}
\]

---

# 4. Full memory architecture

We then incorporated the major components:

- vector write gating
- sparse top-\(k\) routing
- candidate writer
- orthogonal updates
- diversity-related regularization
- memory reader
- gated fusion with GPT-2

The full model produced extremely diverse memory geometry, with effective rank close to the maximum possible rank of 8 and very low pairwise slot cosine.

So from a representation-diversity perspective, the memory looked healthy.

But that raised the next question:

> **Are those diverse slots actually storing and retrieving useful associations?**

That led us to synthetic associative retrieval.

---

# 5. Synthetic associative retrieval task

We created controlled facts such as:

```text
The assigned keyword for Project-000001 is rabbit.
Remember that the keyword associated with Project-000001 is rabbit.
```

Later the model receives:

```text
The assigned keyword for Project-000001 is
```

and should retrieve:

```text
rabbit
```

We used a vocabulary of 16 possible answers.

We also evaluated using candidate answers rather than trusting only normal LM loss.

For the original 4-way forced-choice evaluation:

\[
P(\text{correct by chance})=25\%.
\]

This allowed us to directly test whether the external memory was doing associative retrieval.

---

# 6. First major problem — Loss falls, accuracy doesn't

During synthetic retrieval training, something strange happened.

Retrieval loss dropped dramatically, roughly from:

\[
12.0\rightarrow2.8
\]

but retrieval accuracy stayed around chance.

Results across different distractor distances were around:

```text
25%
22%
25%
21%
19%
```

with 25% being chance for the four-way test.

### Initial diagnosis

We considered two major possibilities.

### Possibility A — Shortcut learning

The model might be learning:

- answer frequencies
- prompt structure
- token priors
- punctuation/template patterns

rather than actually retrieving the stored entity→answer association.

### Possibility B — Broken gradient flow

The fact was written into memory at an earlier step, while retrieval loss occurred later.

If memory was detached between these operations:

\[
L_{\text{retrieval}}
\not\rightarrow
W_{\text{writer/router}}
\]

then the writer could never learn:

> “I should have stored this information because it was needed later.”

This motivated our gradient-flow diagnosis.

---

# 7. Gradient-flow fix — `keep_graph=True`

We found that the training pipeline had been detaching the memory graph between writes.

For training, we changed the context processing to preserve the computational graph:

```text
keep_graph=True
```

while validation can still use:

```text
keep_graph=False
```

This allowed the later retrieval loss to backpropagate through earlier memory writes.

We retrained using 500 training examples and 100 validation examples.

Results:

| Stage | Retrieval accuracy |
|---|---:|
| Before training | 24% |
| Epoch 1 | **31%** |
| Epoch 2 | 24% |
| Epoch 3 | 29% |

Loss dropped:

\[
12.0657\rightarrow3.6299.
\]

So fixing gradient flow was necessary, but it **did not solve retrieval**.

The accuracy remained close to chance.

---

# 8. Memory-dependence diagnosis

Next we asked a very basic question:

> Is prediction actually dependent on the correct memory?

We compared:

```text
Correct memory
No memory
Wrong/shuffled memory
```

Results:

| Condition | Accuracy |
|---|---:|
| Correct memory | 31% |
| No memory | 28% |
| Wrong memory | 30% |

### Interpretation

This was a major red flag.

If associative retrieval were working, correct memory should substantially outperform wrong memory.

Instead:

\[
31\%\approx30\%\approx28\%.
\]

Therefore:

> **The model was using the existence/state of memory somewhat, but was largely insensitive to whether that memory contained the correct fact.**

That motivated a layer-by-layer diagnosis.

---

# 9. Correct-memory vs wrong-memory diagnostic

We created two different facts, such as:

```text
Project-100000 → rabbit
Project-100001 → river
```

and compared what happened throughout the architecture.

We inspected:

1. router
2. writer
3. orthogonal update
4. write gate
5. final memory
6. reader
7. fusion
8. logits

---

# 10. Router diagnosis

For rabbit vs river, the router outputs were almost identical.

Router-weight cosine:

\[
\boxed{0.999999}
\]

Both facts selected:

```text
[7, 2]
```

### Interpretation

Different facts were being sent to essentially the **same memory addresses**.

So the router wasn't behaving like:

```text
Fact A → one address
Fact B → another address
```

Instead:

```text
Fact A → {7,2}
Fact B → {7,2}
```

This was our first clear indication of **write-address collapse**.

---

# 11. Writer diagnosis

The writer's attended token context did retain some difference between the two facts.

Its relative difference was about:

\[
15.5\%.
\]

So the writer was not completely blind to the different sentences.

But after the candidate-writing transformation, candidate vectors became extremely similar:

\[
\cos(C_A,C_B)\approx0.999808.
\]

Writer deltas were also highly similar:

\[
\cos(\Delta_A,\Delta_B)\approx0.999456.
\]

### Interpretation

Some fact-specific information existed at the writer input, but much of it was compressed away before becoming a memory update.

So another part of the failure existed on the write side.

---

# 12. Final memory diagnosis

After writing rabbit vs river, the resulting complete memories had:

\[
\boxed{\cos(M_A,M_B)=0.999819}
\]

Only slots 2 and 7 changed meaningfully because both facts had been routed there.

### Interpretation

The external memory after writing two semantically different facts was almost identical.

Therefore:

> The model was not producing sufficiently fact-specific memory states.

---

# 13. Reader diagnosis

The problem became even worse on reading.

Correct-memory vs wrong-memory reader context:

\[
\cos(R_A,R_B)\approx0.999991.
\]

Reader attention was almost identical.

After fusion:

\[
\cos(H_A,H_B)\approx1.
\]

And the final logits were essentially identical.

The KL divergence between predictions from correct and wrong memories was only about:

\[
5.94\times10^{-6}.
\]

### Interpretation

Even the small difference surviving the writer was effectively ignored by the reader/fusion pathway.

So at this stage our diagnosis was:

\[
\boxed{
\text{Fact discrimination collapses during write}
+
\text{reader further suppresses the remaining distinction}
}
\]

---

# 14. Gradient diagnosis

We then directly checked whether retrieval loss was reaching the memory modules.

Example gradient-norm sums included approximately:

```text
memory bank       → 3.64
router            → 1.32
writer            → 1.85
reader            → 40.55
write gate        → 0.008
```

### Interpretation

This was very important.

The router and writer **were receiving gradients**.

Therefore the remaining failure could no longer simply be explained as:

> “The gradient is detached.”

The graph fix worked.

Instead, the system was receiving learning signal but finding a bad solution.

The reader also dominated numerically, while the write gate received a tiny gradient.

---

# 15. All-16-answer ranking diagnosis

We stopped relying only on four candidates and ranked all 16 possible answers.

For one diagnostic example, the correct answer `rabbit` ranked:

\[
\boxed{16/16}
\]

under several memory conditions.

The model strongly preferred tokens such as `blue`.

### Interpretation

This provided strong evidence of **answer/token-prior learning**.

The decreasing LM loss did not mean:

> “The model learned Project-X → rabbit.”

It could reduce loss by learning properties of the tiny answer distribution and template.

This confirmed:

\[
\boxed{\text{Low LM loss}\neq\text{successful associative retrieval}}
\]

---

# 16. Two-fact fresh-memory overfit test

Next we simplified the problem drastically.

We used:

```text
A → rabbit
B → river
```

but each was given a **fresh memory**.

After 300 training steps:

```text
A → rabbit rank 1
B → river  rank 1
```

Memory cosine also dropped substantially.

### Interpretation

This was our first important positive control.

It showed:

> **The architecture can learn individual associations when they do not have to coexist in the same memory state.**

However, this did **not** prove simultaneous memory capacity.

Each association got a fresh memory.

---

# 17. N=64 fresh-memory experiment

We expanded the association-learning test up to 64 examples.

At \(N=64\):

```text
Accuracy = 93.75%
MRR = 0.9609
Mean rank = 1.11
```

This initially looked very strong.

But each example again began with:

```python
memory_state=None
```

### Interpretation

Therefore this was not really a 64-fact simultaneous capacity test.

It demonstrated:

> The model can learn many entity→answer mappings when each example receives a fresh external memory.

It did **not** demonstrate:

> The memory can simultaneously hold 64 facts.

This distinction became crucial.

---

# 18. Sequential multi-fact memory test

We then actually wrote multiple facts sequentially into the **same memory**.

Results:

| Facts in same memory | Accuracy |
|---:|---:|
| 1 | **100%** |
| 2 | **0%** |
| 4 | 0% |
| 8 | 12.5% |
| 12 | 8.33% |
| 16 | 6.25% |

For 16 candidates:

\[
\text{chance}=6.25\%.
\]

### Interpretation

This exposed the real problem.

The model worked beautifully with one fact.

As soon as another fact was written into the same memory:

\[
100\%\rightarrow0\%.
\]

Therefore the key failure was **multi-fact coexistence/interference**.

---

# 19. Two-fact interference diagnosis

We then looked at exactly what the second fact did.

Example:

```text
A = Project-0000 → tiger
B = Project-0001 → apple
```

A alone:

```text
tiger rank 1
```

B alone:

```text
apple rank 1
```

So both facts worked individually.

But when B was written after A, both used:

```text
slots [7,2]
```

B changed those slots heavily.

After A+B:

```text
A → tiger rank 7
B → apple rank 14
```

### Interpretation

This ruled out a simple “model only remembers the latest fact” explanation.

The second fact didn't merely replace A with B.

Instead, both associations became corrupted.

We therefore identified:

\[
\boxed{\text{memory-slot collision/interference}}
\]

---

# 20. Sequential two-fact training

Next we explicitly trained:

```text
write A
↓
write B
↓
query A from final memory
query B from final memory
```

The model was therefore directly told:

> Both facts must survive simultaneously.

But training oscillated.

Sometimes both queries preferred `tiger`; later both preferred `apple`.

### Interpretation

The model learned:

> “These two answer tokens are important.”

but not:

\[
Q_A\rightarrow A
\]

and

\[
Q_B\rightarrow B.
\]

That suggested a **binding problem**, rather than simply insufficient retrieval supervision.

---

# 21. Query-discrimination diagnosis

We then checked whether GPT-2 itself could distinguish:

```text
Query A
```

from:

```text
Query B.
```

The base GPT-2 hidden representations were different.

But after the memory reader:

```text
Reader attention A ≈ Reader attention B
Reader context A ≈ Reader context B
```

Reader cosine was essentially:

\[
1.0.
\]

### Interpretation

This was extremely important.

GPT-2 **had entity/query information**.

But the reader discarded that distinction and mapped both queries to the same memory retrieval.

So we identified:

\[
\boxed{\text{read-addressing collapse}}
\]

---

# 22. Sparse collision-loss experiment

We then explicitly penalized A and B for using the same write routes.

The loss encouraged:

\[
r_A\neq r_B.
\]

The routes changed somewhat in proportion:

```text
A → more slot 2
B → more slot 7
```

but both still used the same support:

```text
{2,7}
```

Final retrieval still produced only one surviving association.

### Interpretation

The collision objective helped somewhat but could not produce genuinely separate memory addresses.

---

# 23. Token-reader experiment

We changed the reader from:

```text
hybrid
```

to:

```text
token
```

to preserve more query-specific information.

Write-route proportions became more different.

However, the reader still produced essentially identical memory attention for A and B.

The reader-context relative difference was only around:

\[
10^{-5}.
\]

### Interpretation

The hybrid summary mechanism was **not the primary cause**.

Even token-level queries were being collapsed by the learned reader addressing.

---

# 24. Explicit write→read binding loss

We then introduced a binding objective.

If the writer used route:

\[
r_A
\]

then Query A's reader should reproduce:

\[
r_A.
\]

Likewise:

\[
r_B\rightarrow\text{reader}(Q_B).
\]

The binding loss decreased successfully.

### What happened?

The reader did learn to follow the writer.

But the writer routes themselves were still nearly identical.

So we got:

\[
r_A\approx r_B
\]

therefore:

\[
\text{read}(A)\approx\text{read}(B).
\]

### Interpretation

The binding objective was doing what we asked.

The deeper problem was:

\[
\boxed{\text{writer addresses themselves are not sufficiently distinct}}
\]

This shifted attention back to the router.

---

# 25. Dense pre-top-k collision experiment

We suspected sparse top-\(k\) routing might prevent useful gradients from moving a fact into a completely different slot.

So we penalized overlap on the dense router distribution **before top-k**.

Initially the collision loss decreased.

But the router discovered a loophole.

Both distributions became almost uniform:

\[
r_A\approx
\left[\frac18,\frac18,\ldots,\frac18\right]
\]

\[
r_B\approx
\left[\frac18,\frac18,\ldots,\frac18\right].
\]

For identical uniform distributions:

\[
r_A^\top r_B=\frac18=0.125.
\]

So it could reduce the overlap objective without actually making A and B different.

### Interpretation

This was an **objective-design failure**.

Low dot-product overlap did not necessarily mean distinct routes.

The router exploited the loss by becoming diffuse.

---

# 26. Soft routing + collision + entropy + binding

To prevent the uniform-routing loophole, we removed router top-k during training and added entropy minimization.

The objective became roughly:

\[
L=
L_{\text{retrieval}}
+
L_{\text{collision}}
+
0.5L_{\text{entropy}}
+
L_{\text{binding}}.
\]

Entropy minimization should encourage sharp distributions.

And it worked.

Routing entropy dropped from roughly:

\[
1.61\rightarrow0.71.
\]

But both routes became sharp in the **same places**:

```text
A:
slot 2 = 0.467
slot 7 = 0.530

B:
slot 2 = 0.462
slot 7 = 0.536
```

Final write-route cosine:

\[
\boxed{0.999943}
\]

Reader-route cosine:

\[
\boxed{0.999972}.
\]

The final retrieval remained:

```text
A rabbit → river, rank 2
B river  → river, rank 1
```

This run started from the retrieval diagnostic checkpoint, used token reader mode, disabled router top-k during the experiment, and used collision, entropy, and binding objectives. Pasted text

### Interpretation

Entropy solved the **diffuse routing** problem.

But it did not solve the **fact-specific allocation** problem.

Instead of:

\[
A\rightarrow M_2,\qquad B\rightarrow M_7
\]

we still got:

\[
A\rightarrow\{M_2,M_7\}
\]

\[
B\rightarrow\{M_2,M_7\}.
\]

So simply changing loss weights was no longer looking promising.

---

# 27. Forced-write diagnosis

This was our first decisive isolation experiment.

We completely bypassed learned write routing.

We forced:

\[
A\rightarrow M_2
\]

\[
B\rightarrow M_7.
\]

Now we asked the normal learned reader to retrieve them.

After training:

```text
Query A:
slot 2 = 62.4%
slot 7 = 37.6%

Query B:
slot 2 = 25.1%
slot 7 = 74.9%
```

Reader cosine dropped to:

\[
0.7605.
\]

This was dramatically better than the previous \(\approx1.0\).

### Interpretation

This told us two things.

First:

> **Separating the writes helps enormously.**

The reader is capable of distinguishing the queries to some extent when the facts actually occupy different locations.

Second:

> **The learned reader is still insufficiently selective.**

A should ideally retrieve:

\[
[0,0,1,0,0,0,0,0]
\]

but instead it still took 37.6% from B's slot.

This contamination was enough that both final predictions still leaned toward `river`.

So we now had evidence for **two addressing problems**:

\[
\boxed{\text{write-address allocation failure}}
\]

and

\[
\boxed{\text{read-address selectivity failure}}.
\]

---

# 28. Forced-write + forced-read experiment

This was the most important diagnostic so far.

We removed learned addressing completely.

We forced:

\[
A\rightarrow M_2
\]

\[
B\rightarrow M_7
\]

during writing.

Then during retrieval:

\[
Q_A\rightarrow M_2
\]

\[
Q_B\rightarrow M_7.
\]

Reader attention was exactly:

```text
A:
[0,0,1,0,0,0,0,0]

B:
[0,0,0,0,0,0,0,1]
```

Then the result was:

| Query | Correct answer | Prediction | Rank | Loss |
|---|---|---|---:|---:|
| A | rabbit | **rabbit** | **1/16** | **0.0004** |
| B | river | **river** | **1/16** | **0.0072** |

By only 25 steps, both were already rank 1.

---

# 29. What the final forced experiment proves

This gives us the cleanest diagnosis so far.

When addressing is correct:

\[
\boxed{
A\rightarrow M_2\rightarrow rabbit
}
\]

and

\[
\boxed{
B\rightarrow M_7\rightarrow river
}
\]

work simultaneously.

Therefore, in this controlled two-association overfit setting, we do **not** have evidence that the core memory representation is fundamentally incapable of holding multiple associations.

The following pathway can work:

```text
Writer
  ↓
Memory content
  ↓
Reader value
  ↓
Fusion
  ↓
GPT-2
  ↓
Correct answer
```

provided we give it the correct memory address.

Our main experimentally isolated failure is therefore:

# **MEMORY ADDRESSING**

More specifically:

\[
\boxed{
\text{Learned write allocation}
+
\text{Learned query-specific read selection}
}
\]

---

# 30. Complete diagnosis chain

The entire debugging process can now be summarized as:

```text
Loss decreases but retrieval ≈ chance
                ↓
Is gradient disconnected?
                ↓
YES partially → fixed keep_graph=True
                ↓
Still ≈ chance
                ↓
Does correct memory matter?
                ↓
Barely
                ↓
Compare correct vs wrong memory internally
                ↓
Router sends different facts to same slots
                ↓
Writer produces highly similar memory states
                ↓
Reader gives almost identical retrieval
                ↓
Test facts individually
                ↓
Each fact works alone
                ↓
Put two facts in same memory
                ↓
Catastrophic interference
                ↓
Train sequentially
                ↓
Still no A↔rabbit / B↔river binding
                ↓
Inspect query representations
                ↓
GPT-2 distinguishes A and B
but reader retrieves same thing
                ↓
Add collision loss
                ↓
Same slots, different proportions
                ↓
Use token reader
                ↓
Still read collapse
                ↓
Add explicit write-read binding
                ↓
Reader follows writer,
but writer addresses are same
                ↓
Dense collision
                ↓
Router exploits loss by becoming uniform
                ↓
Add entropy/sharpness
                ↓
Sharp routes, but SAME sharp routes
                ↓
Force A and B into different write slots
                ↓
Reader finally distinguishes them partially
                ↓
But cross-talk remains
                ↓
Force write AND read addresses
                ↓
rabbit rank 1 ✓
river rank 1 ✓
                ↓
MAIN BOTTLENECK ISOLATED:
LEARNED MEMORY ADDRESSING
```

---

# 31. What we should NOT conclude

There are several claims we need to avoid.

We should **not** say:

> “The entire memory architecture works perfectly.”

We tested a tiny two-association overfitting case.

We should instead say:

> **Under a controlled two-association experiment, the system successfully stores and retrieves both associations when their write and read addresses are explicitly separated. This suggests that the observed multi-fact retrieval failure is primarily associated with learned memory addressing rather than an inherent inability of the memory representation and fusion pathway to support multiple associations.**

That is much stronger scientifically because it matches exactly what we tested.

---

# 32. What are we planning now?

At this point I would stop tuning things like:

```text
collision weight = 0.5 vs 1
entropy weight = 0.5 vs 1
top-k = 2 vs soft
```

We've extracted enough information from those experiments.

The next phase should redesign **addressing**.

## Proposed direction — Key–Value, occupancy-aware memory

Currently, the router behaves approximately like:

\[
r=\operatorname{softmax}(W_r h)
\]

so it largely asks:

> Based on this fact representation, which slot do I usually prefer?

What it needs to ask is:

> What does each slot already contain, and should I update an existing related slot or allocate a new one?

So we move toward:

\[
M_i=(K_i,V_i)
\]

where:

- \(K_i\) = memory key/address
- \(V_i\) = stored content/value

### Writing

From the incoming fact:

\[
q_w=W_wh_{\text{fact}}.
\]

Compare \(q_w\) against existing keys:

\[
s_i=\operatorname{sim}(q_w,K_i).
\]

Also maintain occupancy/utilization:

\[
O_i.
\]

Then the system decides between:

**Existing matching slot**

\[
\text{high similarity}\Rightarrow\text{update it}
\]

and:

**New unrelated fact**

\[
\text{low similarity}\Rightarrow\text{allocate a free/least-used slot}.
\]

Conceptually:

```text
Project-A → rabbit
        ↓
Is Project-A already represented?
        ↓
NO
        ↓
Allocate slot 2
        ↓
K2 = Project-A representation
V2 = rabbit/fact representation


Project-B → river
        ↓
Does it match slot 2?
        ↓
NO
        ↓
Slot 2 already occupied
        ↓
Allocate slot 7
        ↓
K7 = Project-B representation
V7 = river/fact representation
```

---

# 33. Reading under the new design

For a query:

```text
What is the keyword for Project-A?
```

produce:

\[
q_r=W_qh_{\text{query}}.
\]

Then retrieve based primarily on the **keys**:

\[
a_i=
\frac{q_r^\top K_i}{\sqrt d}.
\]

\[
\alpha_i=\operatorname{softmax}(a_i).
\]

Then:

\[
m=\sum_i\alpha_iV_i.
\]

Ideally:

\[
q_A\approx K_A
\]

therefore:

\[
\alpha_A\approx1.
\]

So:

\[
Q_A
\rightarrow K_A
\rightarrow V_A
\rightarrow rabbit.
\]

Likewise:

\[
Q_B
\rightarrow K_B
\rightarrow V_B
\rightarrow river.
\]

This explicitly introduces the relationship we were missing:

\[
\boxed{
\text{Entity/query}
\leftrightarrow
\text{memory address/key}
\leftrightarrow
\text{stored value}
}
\]

---

# 34. What the next experiments should be

After implementing the new addressing mechanism, we should **not immediately jump to WikiText or LoCoMo**.

We should repeat our diagnostic ladder.

First:

```text
A → rabbit
B → river
```

with fully learned addressing.

We want:

\[
A\rightarrow M_i
\]

\[
B\rightarrow M_j,\quad i\neq j
\]

without manually forcing the slots.

Then query:

\[
Q_A\rightarrow M_i
\]

\[
Q_B\rightarrow M_j.
\]

Both should be rank 1.

Then increase:

\[
2\rightarrow4\rightarrow8
\]

simultaneous facts.

Only once that works do we move toward:

- distractor distance
- longer retention
- randomized entity-answer mappings
- multiple seeds
- RULER
- PG-19
- LoCoMo
- latency/throughput and memory-utilization evaluation.

---

# 35. Where the project stands right now

The project has actually progressed from:

> **“Retrieval accuracy is bad and we don't know why.”**

to a much more specific research finding:

> **The model can learn individual associations and can simultaneously store/retrieve two associations when their addresses are explicitly separated. However, its learned addressing mechanism fails to reliably allocate distinct facts to distinct memory locations and to selectively retrieve the corresponding location from a query.**

That is our current central technical problem:

\[
\boxed{
\textbf{WRITE–READ ADDRESS BINDING / MEMORY ADDRESSING COLLAPSE}
}
\]

And our next architectural objective is:

\[
\boxed{
\textbf{Learn content-aware, occupancy-aware key–value addressing}
}
\]

rather than continuing to try to fix the same router using progressively more loss penalties.