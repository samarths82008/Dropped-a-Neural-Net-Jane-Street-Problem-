# I Dropped a Neural Net — Jane Street Puzzle Solution

> **Puzzle:** [Jane Street — I Dropped a Neural Net](https://huggingface.co/spaces/jane-street/droppedaneuralnet)  
> **Result:** ✅ Solved — MSE = 0.000000 on full 10,000-sample dataset  
> **Runtime:** Under 3 minutes on CPU. No GPU. No gradient descent.

---

## The Problem

A 48-block residual neural network was "dropped" and its 97 weight files got shuffled. Given nothing but the scrambled weights and 10,000 input/output pairs from the original network, reconstruct the exact original ordering.

The network architecture:
```python
class Block(nn.Module):
    def forward(self, x):
        residual = x
        x = self.inp(x)        # Linear(48 -> 96)
        x = self.activation(x) # ReLU
        x = self.out(x)        # Linear(96 -> 48)
        return residual + x    # residual connection

class LastLayer(nn.Module):
    def forward(self, x):
        return self.layer(x)   # Linear(48 -> 1)
```

The 97 pieces break down as:
- **48 type-A pieces** — `inp` layers, weight shape `(96, 48)`
- **48 type-B pieces** — `out` layers, weight shape `(48, 96)`  
- **1 type-C piece** — `LastLayer`, weight shape `(1, 48)` → always `piece_85`

The solution is a permutation of 0–96 where even positions hold inp layers, odd positions hold out layers, and position 96 holds the last layer.

The combined search space: **(48!)² ≈ 10¹²²** — more combinations than atoms in the observable universe. Brute force is impossible.

---

## The Two Sub-Problems

**Pairing:** Which of the 48 inp layers belongs with which of the 48 out layers to form each block? (48! possibilities)

**Ordering:** In what sequence do the 48 paired blocks execute? (48! possibilities)

---

## What Didn't Work

Before finding the solution I tried several approaches that failed:

### Greedy Sequential Construction
Build the network one block at a time, always picking whichever remaining block minimises the partial MSE. Fast to implement but gives MSE ~1.8 — worse than predicting the mean. The problem is myopia: a block that looks optimal at step k reshapes the hidden state in a way that makes every subsequent block harder to place. Early mistakes compound through all 48 layers with no recovery.

### Simulated Annealing
Random swaps of block positions, pairing swaps, insertions, segment reversals with a temperature schedule. Each evaluation requires a full forward pass through 48 blocks on 10,000 samples (~7ms per eval). At 500K steps that's nearly an hour per run, and SA needs many restarts to explore a space this large. Every improvement it found was something deterministic local search would have found faster.

### Gumbel-Sinkhorn Gradient Descent
Relax the permutation to a differentiable doubly-stochastic matrix using Sinkhorn normalisation. Optimise with Adam, alternating between fixing pairing and optimising ordering. This got to MSE ~0.12 after 6 alternations on GPU but took 45+ minutes and the local search phase still couldn't close the gap to zero. Kept hitting the same local minima.

**The core problem with all of these:** pairing was wrong. If even one block is matched to the wrong output projection, no amount of ordering search can compensate — the block produces wrong intermediate values and the entire network is poisoned.

---

## What Actually Worked

### Step 1 — Pairing via Diagonal Dominance Ratio

The key insight comes from **dynamic isometry** — the mathematical property that well-trained residual networks must satisfy for stable gradient flow during training.

For a residual block with Jacobian `J = I + W_out · D · W_in` (where D is the ReLU gating matrix), the isometry condition `E[J^T J] = I` expands to:

```
2 · E[tr(W_out · D · W_in)] + E[||W_out · D · W_in||²] = 0
```

Since the squared norm term is always ≥ 0, the trace term must be negative. With ReLU firing ~50% of the time, `E[D] = ½I`, so:

```
tr(W_out · W_in) < 0    ← always true for correctly paired blocks
```

Training burns a **negative diagonal structure** into the product matrix of every correct pair. This gives us a clean signal:

**Diagonal Dominance Ratio:**
```
d(i, j) = |tr(W_out_j · W_in_i)| / ||W_out_j · W_in_i||_F
```

| Pair type | Score range |
|-----------|-------------|
| ✅ Correct pairs | 1.76 – 3.28 |
| ❌ Wrong pairs | 0.00 – 0.58 |

**Gap of 1.18. Zero overlap. Perfect separation.**

Compute `d(i,j)` for all 48×48 = 2,304 candidates, then use the Hungarian algorithm to find the assignment maximising total score. Runs in milliseconds. No data needed — the signal is entirely in the weight structure.

```python
# Compute diagonal dominance matrix
D = np.zeros((48, 48))
for i in range(48):
    for j in range(48):
        M = wB[j] @ wA[i]                          # W_out · W_in → (48,48)
        D[i,j] = np.abs(np.trace(M)) / np.linalg.norm(M, 'fro')

# Solve assignment (maximise)
_, col = linear_sum_assignment(-D)
pairing = list(col)    # pairing[i] = j means inp_i pairs with out_j
```

### Step 2 — Ordering via Frobenius Norm Seed + Bubble Sort

**Seed:** In trained residual networks, deeper blocks make larger corrections to the hidden state and tend to have larger output projection norms. Sorting blocks by ascending `||W_out||_F` gives a rough depth ordering.

This data-free heuristic seeds at **MSE ≈ 0.076** instantly.

**Hill-climbing:** Sweep through all 47 adjacent block pairs. Swap if MSE decreases, undo if not. Repeat until a full sweep makes zero swaps.

```python
# Seed: sort by ||Wout||_F ascending
order = sorted(range(48), key=lambda i: np.linalg.norm(wB[pairing[i]], 'fro'))

# Bubble-sort hill-climbing
while True:
    swaps = 0
    for k in range(47):
        order[k], order[k+1] = order[k+1], order[k]
        if fwd_mse(order, pairing) < current_mse:
            current_mse = fwd_mse(order, pairing)
            swaps += 1
        else:
            order[k], order[k+1] = order[k+1], order[k]  # undo
    if swaps == 0:
        break
```

**Convergence:**

| Round | Swaps | MSE |
|-------|-------|-----|
| Seed  | —     | 0.075851 |
| 1     | 30    | 0.005936 |
| 2     | 16    | 0.003397 |
| 3     | 12    | 0.001405 |
| 4     | 9     | 0.000568 |
| 5     | 4     | 0.000223 |
| 6     | 1     | **0.000000** ✅ |

6 rounds, 72 total swaps, exact solution.

Note: Hyunwoo Park's paper uses Bradley-Terry ranking as an intermediate step (reducing convergence from 13 rounds to 5). I skipped it and went straight from the Frobenius seed to bubble repair — with a good seed it converges in 6 rounds anyway.

---

## Results

```
Pairing scores:  min=1.7639  max=3.2320  mean=2.7849
Wrong pair max:  0.5838
Seed MSE:        0.075851
Final MSE:       0.000000000000
Total time:      ~3 minutes on CPU
```

---

## How to Run

```bash
# Requirements
pip install numpy scipy

# Place these in the same directory:
# - historical_data.csv
# - pieces/ (folder with piece_0.pth ... piece_96.pth)
# - solve.py

python solve.py
# Prints solution and saves to solution.txt
```

---

## Files

| File | Description |
|------|-------------|
| `solve.py` | Complete solution — pairing + ordering |
| `solution.txt` | The 97-number permutation |
| `explanation.html` | Full visual walkthrough of every step |

---

## Credit

The diagonal dominance insight is from:

> Hyunwoo Park, "I Dropped a Neural Net", Carnegie Mellon University, arXiv:2602.19845 (2026)

---

## About

Samarth Singh — high school senior from San Jose, CA, heading to UC Santa Cruz to study Applied Math.  
[LinkedIn](https://linkedin.com/in/yourprofile) · [Email](mailto:samarths82008@gmail.com)
