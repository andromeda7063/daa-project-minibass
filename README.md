# MiniBass: Low-Frequency Sound Field Optimization and Synthesis

**MiniBass** is a high-performance acoustic optimization engine and mathematical modeling library for multi-subwoofer sound field synthesis in enclosed rectangular acoustic spaces.

It couples first-principles 3D Helmholtz modal Green's functions, 4th-order Thiele-Small vented electro-acoustic transducers, discrete active DSP control policies, and vectorized combinatorial search algorithms.

---

## Key Features & Highlights

- **First-Principles Modal Acoustics:** Exact 3D rectangular cavity Green's function with modal damping ($\delta_n = 6.91 / T_{60}$) and Neumann normalization factors, avoiding empirical interpolation artifacts.
- **Transducer Electromechanics:** Full 4th-order Thiele-Small coupled matrix solver evaluating voice-coil electrical dynamics, acoustic box compliance, and port air mass resonance.
- **The Optimization Algorithms:**
  0. `LayeredBeamSearchOptimizer` (`search_instance`): Memory-bounded discrete beam-search algorithm from Section 3 of the research report ($W=32$, canonical position ordering, Pareto filtering).
  1. `VectorizedGeometricScanner`: Tensor SIMD evaluator processing $>50,000$ combinations/sec using precomputed complex acoustic atoms.
  2. `MultiSeatMinimaxOptimizer`: Audience fairness optimizer maximizing worst-seat passing frequency count and computing the seat conjunction spectrum $N(k)$.
  3. `RLTActiveBilinearSolver`: Reformulation-Linearization Technique solver isolating pairwise destructive interference terms for active boundary absorption / dipole synthesis.
- **Rigorous Verification & Generalization:**
  - In-sample (80 seats) and held-out off-grid (40 seats) validation verifying true spatial smoothing without grid-point overfitting ($6.25\text{ dB}$ in-sample vs. $6.16\text{ dB}$ held-out RMS error).
  - 256-sample environmental and physical tolerance Monte Carlo ($c \pm 2\%$, $T_{60} \pm 10\%$, speaker placement $\sigma = 5\text{ cm}$, seat movement $\sigma = 10\text{ cm}$, driver parameters $\pm 5\%$).
  - Machine-checked Lean 4 formal certification of single-frequency projection bounds and lexicographic order preservation (0 axioms, 0 sorry).

---

## Optimization Algorithms

All algorithms are consolidated in [`minibass/algorithms.py`](file:///home/andromeda/Documents/code/projects/spok/minibass/algorithms.py).

### 0. Layered Beam Search Solver (`LayeredBeamSearchOptimizer` / `search_instance`)
- **Principle:** The exact discrete beam search solver presented in Section 3 of the research report.
- **Mechanism:** Explores multi-source combinations across depths $d = 1 \dots 4$. Uses canonical position ordering $\operatorname{pos}(a) > \operatorname{pos}(a_{\text{last}})$ to eliminate permutation redundancy, accumulates complex pressures $P(r, k)$ in SIMD batches, filters candidates via Pareto non-dominated sorting, and retains the top $W = 32$ partial candidates using 16 multi-objective scalarization weights.
- **Throughput:** Evaluated 104,145 combinations in 22.4 seconds ($4,649\text{ candidates/s}$) during the $10 \times 12\text{ m}$ venue synthesis.

### 1. Vectorized Geometric Tensor Scanner (`VectorizedGeometricScanner`)
- **Principle:** Decouples room transfer functions from transducer and DSP parameters. For each discrete atom $a = (\text{position } \mathbf{x}_s, \text{enclosure/driver } d, \text{active DSP } c)$, the complex pressure tensor at receiver $r$ and frequency $k$ is precomputed:
  $$A(a, r, k) = H(r, \mathbf{x}_s, k) \cdot q_a(k)$$
- **Broadcasting:** Multi-source evaluation over a 4-subwoofer layout is vectorized via chunked SIMD tensor addition:
  $$P(r, k) = \sum_{i=1}^4 A(a_i, r, k)$$
- **Performance:** Evaluates $>50,000$ configurations per second, scoring cell pass rate within $\pm 3.0\text{ dB}$ and room-wide RMS flatness error.

### 2. Multi-Seat Minimax Optimizer (`MultiSeatMinimaxOptimizer`)
- **Principle:** Standard mean-error objectives allow large localized acoustic nulls if other seats are slightly improved. This optimizer enforces audience fairness by solving the minimax problem:
  $$\max_{C} \min_{s} \sum_{k=1}^K \mathbb{I}(p_{\text{lower}} \le |P(s, k)| \le p_{\text{upper}})$$
- **Conjunction Spectrum $N(k)$:** Evaluates the cumulative audience coverage function:
  $$N(k) = \big| \{s \in \mathcal{S} : \text{passing\_tones}(s) \ge k\} \big|$$
  guaranteeing that worst-case audience seats satisfy acceptable listening criteria.

### 3. RLT Active Bilinear Boundary Solver (`RLTActiveBilinearSolver`)
- **Principle:** Reformulates acoustic sound field power into individual source terms and pairwise bilinear cross-interference terms:
  $$|P(s, f)|^2 = \sum_{i=1}^M |p_i(s, f)|^2 + 2 \sum_{i < j} \operatorname{Re}\big( p_i(s, f) \, p_j^*(s, f) \big)$$
- **Application:** Identifies boundary cancellation pairs that exploit destructive interference ($\operatorname{Re}(p_i p_j^*) < 0$) to absorb longitudinal standing wave reflections (active boundary absorption / dipole synthesis) with minimal enclosure volume.

---

## $10 \times 12\text{ m}$ Performance Venue Dry Run Benchmark

In the standardized $10.0 \times 12.0 \times 2.8\text{ m}$ hall ($V = 336\text{ m}^3$, $T_{60} = 0.45\text{ s}$, centered stage $8.0 \times 2.0\text{ m}$, $0.2\text{ m}$ margins, 80 in-sample seats, 40 held-out off-grid seats, 95 frequencies from 20 to 200 Hz):

| Configuration | In-Sample RMS | Held-Out RMS | In-Sample Worst Seat | Held-Out Worst Seat | In-Sample Pass ($\pm 3\text{ dB}$) | Held-Out Pass ($\pm 3\text{ dB}$) | Mean Power |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Single Sub (Corner, Pos 0) | $9.33\text{ dB}$ | $9.39\text{ dB}$ | $16.11\text{ dB}$ | $14.25\text{ dB}$ | $28.4\%$ | $28.0\%$ | $109.4\text{ W}$ |
| Single Sub (Center, Pos 3) | $9.62\text{ dB}$ | $9.39\text{ dB}$ | $15.61\text{ dB}$ | $14.97\text{ dB}$ | $25.9\%$ | $24.7\%$ | $109.4\text{ W}$ |
| Symmetric 4-Sub (No DSP) | $6.99\text{ dB}$ | $7.27\text{ dB}$ | $11.52\text{ dB}$ | $9.61\text{ dB}$ | $38.9\%$ | $36.1\%$ | $194.5\text{ W}$ |
| Ablation 1: Champ Layout (No DSP) | $6.59\text{ dB}$ | $6.44\text{ dB}$ | $10.84\text{ dB}$ | $9.68\text{ dB}$ | $41.1\%$ | $39.3\%$ | $237.7\text{ W}$ |
| Ablation 2: Champ + Delay Only | $6.28\text{ dB}$ | $6.19\text{ dB}$ | $9.82\text{ dB}$ | $8.62\text{ dB}$ | $38.9\%$ | $36.0\%$ | $337.9\text{ W}$ |
| Ablation 3: Champ + Lowpass Only | $6.45\text{ dB}$ | $6.27\text{ dB}$ | $10.13\text{ dB}$ | $8.83\text{ dB}$ | $38.9\%$ | $37.1\%$ | $298.6\text{ W}$ |
| **Champion Array (Asymm + Delay + LP)** | **$6.25\text{ dB}$** | **$6.16\text{ dB}$** | **$9.97\text{ dB}$** | **$9.80\text{ dB}$** | **$39.3\%$** | **$37.6\%$** | **$3.14\text{ W}$** |

### Champion Array Layout & DSP Parameters
- **Sub 1 ($x = 3.39\text{ m}$, Pos 2):** 10" ported box (`box-1458`, $41.5\text{ L}$ gross, $29.8\text{ L}$ net), driver `s03`, Gain: 3.0, Delay: $2.0\text{ ms}$, Filter: Flat.
- **Sub 2 ($x = 4.46\text{ m}$, Pos 3):** 10" ported box (`box-1458`, driver `s03`), Gain: 3.0, Delay: $2.0\text{ ms}$, Filter: Flat.
- **Sub 3 ($x = 7.68\text{ m}$, Pos 6):** 10" ported box (`box-1458`, driver `s03`), Gain: 3.0, Delay: $0.0\text{ ms}$, Filter: $120\text{ Hz}$ Lowpass.
- **Sub 4 ($x = 8.75\text{ m}$, Pos 7):** 10" ported box (`box-83a0`, driver `s04`), Gain: 1.0, Delay: $0.0\text{ ms}$, Filter: $120\text{ Hz}$ Lowpass.

---

## Physical and Environmental Uncertainty Audit

A 256-sample Monte Carlo sensitivity test perturbs:
- Speed of sound: $c \sim \mathcal{U}(336.1, 349.9)\text{ m/s}$ ($\pm 2\%$ temperature variations)
- Boundary damping: $T_{60} \sim \mathcal{U}(0.405, 0.495)\text{ s}$ ($\pm 10\%$)
- Loudspeaker placement: Gaussian jitter $\sigma = 0.05\text{ m}$ ($5\text{ cm}$)
- Listener position: Gaussian jitter $\sigma = 0.10\text{ m}$ ($10\text{ cm}$)
- Transducer voice-coil & suspension: $\pm 5\%$ variation

**Results:**
- RMS Flatness Error: Mean $6.28\text{ dB}$, 90% Confidence Interval $[6.18, 6.40]\text{ dB}$.
- Cell Pass Rate ($\pm 3.0\text{ dB}$): Mean $39.0\%$, 90% Confidence Interval $[37.5\%, 40.5\%]$.
- Stability: $100\%$ of realizations maintain performance within $0.3\text{ dB}$ of nominal.

---

## Project Structure

```
minibass/
├── __init__.py           # Package exports and version metadata
├── physics.py            # 3D Helmholtz Green's function & 4th-order Thiele-Small solver
├── data_generation.py    # 10x12m venue, driver/enclosure catalog, atom precomputation
├── algorithms.py         # The 3 Best Algorithms (Scanner, Minimax, Bilinear RLT)
└── dryrun_10x12.py       # Standalone dry-run runner and benchmark evaluator
docs/
├── LITERATURE_SURVEY.md  # 10-paper structured literature review (>= 2025 base papers)
├── RUN_GUIDE_10x12.md    # Step-by-step reproduction guide and mathematical derivations
├── REPORT_10x12_DRYRUN.pdf # Compiled 17-page LaTeX research report
├── REPORT_10x12_DRYRUN.tex # LaTeX source document
└── figures/              # High-resolution PNG and SVG publication figures
proofs/
└── MiniBass.lean         # Lean 4 formal certificates (0 axioms, 0 sorry)
main.py                   # Command-line entry point
pyproject.toml            # Build configuration
```

---

## Quickstart & Reproduction

### Prerequisites
Python 3.11 or later with `numpy`, `scipy`, and `matplotlib`:
```bash
pip install -e .
```

### Run the Dry Run & Benchmark
Execute the complete dry run, computing all 7 configurations, held-out seats, and metrics:
```bash
python main.py
```
or directly:
```bash
python -m minibass.dryrun_10x12
```

### Run the Lean 4 Proof Verification
```bash
lean proofs/MiniBass.lean
```

---

## Reference Documents

- **Dry Run Guide:** Detailed mathematical definitions, acoustic physics derivations, and step-by-step instructions: [docs/RUN_GUIDE_10x12.md](file:///home/andromeda/Documents/code/projects/spok/docs/RUN_GUIDE_10x12.md).
- **Literature Survey:** Structured review of 10 primary papers covering modal decomposition, boundary absorption, and transducer sensitivity: [docs/LITERATURE_SURVEY.md](file:///home/andromeda/Documents/code/projects/spok/docs/LITERATURE_SURVEY.md).
- **Full Research Report:** Complete 17-page PDF report with derivations, formal proofs, and 7 publication-grade figures: [docs/REPORT_10x12_DRYRUN.pdf](file:///home/andromeda/Documents/code/projects/spok/docs/REPORT_10x12_DRYRUN.pdf).

