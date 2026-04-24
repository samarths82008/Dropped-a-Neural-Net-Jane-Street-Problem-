"""
Solver for Jane Street 'Dropped a Neural Net'
Based on: "I Dropped a Neural Net" - Hyunwoo Park, CMU (arXiv:2602.19845)

Pipeline:
  Step 1 - Pairing:  Diagonal dominance ratio of Wout @ Win
                     Correct pairs:   d in [1.76, 3.28]
                     Incorrect pairs: d in [0.00, 0.58]
                     Perfect separation, solved via Hungarian algorithm.
                     Runtime: ~1 second. No data needed.

  Step 2 - Ordering: Seed by ascending ||Wout||_F (data-free).
                     Then bubble-sort hill-climb on MSE until no swaps.
                     Converges to MSE=0 in ~6 rounds, ~72 swaps.
                     Runtime: ~2 minutes.

Total expected runtime: under 3 minutes on CPU.
No GPU needed. No gradient descent. Mathematically guaranteed pairing.
"""

import os, csv, zipfile, time
import numpy as np
from scipy.optimize import linear_sum_assignment

DATA_CSV   = 'historical_data.csv'
PIECES_DIR = 'pieces'

def load_piece(path):
    with zipfile.ZipFile(path) as z:
        d0 = z.read('archive/data/0')
        d1 = z.read('archive/data/1')
    n1 = len(d1) // 4
    w = np.frombuffer(d0, dtype='<f4').astype(np.float64)
    b = np.frombuffer(d1, dtype='<f4').astype(np.float64)
    if   n1 == 96: return w.reshape(96, 48), b
    elif n1 == 48: return w.reshape(48, 96), b
    else:          return w.reshape(1,  48), b

print("Loading pieces...", flush=True)
type_A, type_B, last_idx = [], [], None
wA, bA, wB, bB = {}, {}, {}, {}
wL = bL = None

for fname in sorted(os.listdir(PIECES_DIR)):
    if not fname.endswith('.pth'):
        continue
    idx = int(fname.replace('piece_', '').replace('.pth', ''))
    w, b = load_piece(os.path.join(PIECES_DIR, fname))
    if   w.shape == (96, 48): type_A.append(idx); wA[idx]=w; bA[idx]=b
    elif w.shape == (48, 96): type_B.append(idx); wB[idx]=w; bB[idx]=b
    else:                     last_idx=idx;       wL=w;       bL=b

type_A.sort()
type_B.sort()
print(f"  {len(type_A)} A-pieces, {len(type_B)} B-pieces, last=piece_{last_idx}", flush=True)

print("Loading data...", flush=True)
rows = []
with open(DATA_CSV) as f:
    for row in csv.DictReader(f):
        rows.append(([float(row[f'measurement_{i}']) for i in range(48)], float(row['pred'])))
Xall = np.array([r[0] for r in rows], dtype=np.float64)
yall = np.array([r[1] for r in rows], dtype=np.float64)
print(f"  {len(rows)} rows loaded.", flush=True)

def relu(x): return np.maximum(0.0, x)

def fwd_mse(order, pairing, Xb=None, yb=None):
    if Xb is None: Xb, yb = Xall, yall
    h = Xb.copy()
    for k in range(48):
        ai = order[k]
        bi = pairing[ai]
        h = h + relu(h @ wA[type_A[ai]].T + bA[type_A[ai]]) @ wB[type_B[bi]].T + bB[type_B[bi]]
    pred = (h @ wL.T + bL).squeeze()
    return float(np.mean((pred - yb) ** 2))

t0 = time.time()

# ── STEP 1: PAIRING via Diagonal Dominance Ratio ──────────────────────────────
# Paper Section 3: for correctly paired (Win, Wout), dynamic isometry forces
# tr(Wout @ Win) < 0. The ratio |tr(M)| / ||M||_F separates correct from wrong pairs.
print("\n=== Step 1: Pairing via Diagonal Dominance Ratio ===", flush=True)

n = 48
D = np.zeros((n, n))
for i in range(n):
    Wi = wA[type_A[i]]       # (96, 48)
    for j in range(n):
        Wj = wB[type_B[j]]   # (48, 96)
        M  = Wj @ Wi         # (48, 48)
        trace = np.abs(np.trace(M))
        frob  = np.linalg.norm(M, 'fro')
        D[i, j] = trace / frob if frob > 1e-10 else 0.0

print(f"  D matrix: min={D.min():.4f}  max={D.max():.4f}  mean={D.mean():.4f}  t={time.time()-t0:.1f}s", flush=True)

row_ind, col_ind = linear_sum_assignment(-D)
pairing = list(col_ind)

scores = [D[i, pairing[i]] for i in range(n)]
print(f"  Paired scores: min={min(scores):.4f}  max={max(scores):.4f}  mean={np.mean(scores):.4f}", flush=True)
print(f"  Paper: correct d in [1.76,3.28], incorrect d in [0.00,0.58]", flush=True)
if min(scores) < 0.9:
    print(f"  WARNING: low score {min(scores):.4f} — pairing may be wrong!", flush=True)
else:
    print(f"  Pairing looks correct (all scores > 0.9)", flush=True)

# ── STEP 2: ORDERING via ||Wout||_F seed + bubble-sort hill-climbing ──────────
# Paper Section 4.1: sort ascending by ||Wout||_F -> MSE ~0.075
# Paper Section 4.4: bubble repair converges to MSE=0 in 6 rounds, 72 swaps
print("\n=== Step 2: Ordering ===", flush=True)

frob_norms = [(i, np.linalg.norm(wB[type_B[pairing[i]]], 'fro')) for i in range(n)]
order = [i for i, _ in sorted(frob_norms, key=lambda x: x[1])]
seed_mse = fwd_mse(order, pairing)
print(f"  Seed MSE (||Wout||_F sort) = {seed_mse:.6f}  (paper ~0.075851)  t={time.time()-t0:.1f}s", flush=True)

print("\n  Bubble-sort hill-climbing:", flush=True)
cur_mse = seed_mse
total_swaps = 0
for rnd in range(1, 200):
    swaps = 0
    for k in range(47):
        order[k], order[k+1] = order[k+1], order[k]
        m = fwd_mse(order, pairing)
        if m < cur_mse - 1e-12:
            cur_mse = m; swaps += 1; total_swaps += 1
        else:
            order[k], order[k+1] = order[k+1], order[k]
    print(f"  Round {rnd:2d}: swaps={swaps:3d}  MSE={cur_mse:.8f}  cum={total_swaps}  t={time.time()-t0:.1f}s", flush=True)
    if swaps == 0 or cur_mse < 1e-12:
        break

# If not converged, try delta-norm seed as backup
if cur_mse > 1e-6:
    print(f"\n  Not converged. Trying delta-norm seed (paper ~0.036794)...", flush=True)
    X_s = Xall[:2000]
    delta_norms = []
    for i in range(n):
        bi = pairing[i]
        h = relu(X_s @ wA[type_A[i]].T + bA[type_A[i]])
        delta = h @ wB[type_B[bi]].T + bB[type_B[bi]]
        delta_norms.append((i, float(np.mean(np.linalg.norm(delta, axis=1)))))
    order2 = [i for i, _ in sorted(delta_norms, key=lambda x: x[1])]
    seed_mse2 = fwd_mse(order2, pairing)
    print(f"  Delta-norm seed MSE = {seed_mse2:.6f}  t={time.time()-t0:.1f}s", flush=True)

    cur_mse2 = seed_mse2
    total_swaps2 = 0
    for rnd in range(1, 200):
        swaps = 0
        for k in range(47):
            order2[k], order2[k+1] = order2[k+1], order2[k]
            m = fwd_mse(order2, pairing)
            if m < cur_mse2 - 1e-12:
                cur_mse2 = m; swaps += 1; total_swaps2 += 1
            else:
                order2[k], order2[k+1] = order2[k+1], order2[k]
        print(f"  Round {rnd:2d}: swaps={swaps:3d}  MSE={cur_mse2:.8f}  t={time.time()-t0:.1f}s", flush=True)
        if swaps == 0 or cur_mse2 < 1e-12:
            break

    if cur_mse2 < cur_mse:
        order, cur_mse = order2, cur_mse2
        print(f"  Using delta-norm result.", flush=True)

# ── Final verification ─────────────────────────────────────────────────────────
print(f"\n=== Final Verification ===", flush=True)
final_mse = fwd_mse(order, pairing, Xall, yall)
print(f"  Final MSE (10000 samples) = {final_mse:.12f}", flush=True)
print(f"  Total time: {time.time()-t0:.1f}s", flush=True)

# ── Build and save permutation ─────────────────────────────────────────────────
pi = [None] * 97
for k in range(48):
    pi[2 * k]     = type_A[order[k]]
    pi[2 * k + 1] = type_B[pairing[order[k]]]
pi[96] = last_idx

assert None not in pi and sorted(pi) == list(range(97)), "Invalid permutation!"
sol = ','.join(map(str, pi))

print(f"\n{'='*60}")
print("SOLUTION:")
print(sol)
print(f"{'='*60}")

with open('solution.txt', 'w') as f:
    f.write(sol + '\n')
print("\nSaved to solution.txt — paste into the Jane Street puzzle answer box.")
