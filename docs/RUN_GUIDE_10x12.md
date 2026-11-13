# MiniBass: 10 × 12 m Performance Venue Dry Run Guide

This guide provides instructions to execute, reproduce, and evaluate the acoustic optimization results for the **10.0 × 12.0 × 2.8 m performance venue**.

---

## 1. Physical Venue Specification

- **Room Geometry:** Length $L_x = 10.0\text{ m}$ (short edge), $L_y = 12.0\text{ m}$ (long edge), height $L_z = 2.8\text{ m}$. Enclosed volume $V = 336.0\text{ m}^3$. Reverberation time $T_{60} = 0.45\text{ s}$.
- **Stage Footprint:** Centered along the short front wall ($y = 0$), dimensions $8.0\text{ m}$ (width) $\times 2.0\text{ m}$ (depth), occupying $x \in [1.0, 9.0]\text{ m}$ and $y \in [0.0, 2.0]\text{ m}$.
- **Candidate Subwoofer Menu:** 8 candidate positions along the front stage boundary line ($y = 1.78\text{ m}, z = 0.40\text{ m}$), spanning $x \in [1.25, 8.75]\text{ m}$ in $1.07\text{ m}$ increments.
- **Audience Seating Area:** Constrained strictly by a **0.2 m margin** from the stage front ($y \ge 2.2\text{ m}$) and perimeter walls ($x \in [0.2, 9.8]\text{ m}, y \le 11.8\text{ m}$), forming a $9.6 \times 9.6\text{ m}$ square area.
- **Receiver Sets:**
  - **In-Sample Grid (80 seats):** Regular $10\text{ columns} \times 8\text{ rows}$ grid cell-centered at seated ear height $z = 1.10\text{ m}$ ($\Delta x = 0.96\text{ m}, \Delta y = 1.20\text{ m}$).
  - **Held-Out Validation Set (40 seats):** $8\text{ columns} \times 5\text{ rows}$ placed off-grid between in-sample seats to verify continuous physical generalization.
- **Target SPL & Bandwidth:** $80.0\text{ dB SPL}$ ($0.20\text{ Pa RMS}$) target across 95 refined discrete frequencies from $20\text{ to }200\text{ Hz}$, with a $\pm 3.0\text{ dB}$ ($[77.0, 83.0]\text{ dB SPL}$) tolerance corridor.

---

## 2. Quickstart Execution

To run the complete benchmark and dry run:

```bash
# Direct execution via main entry point
python main.py

# Or run the module directly
python -m minibass.dryrun_10x12
```

Execution completes in approximately **10–15 seconds**, evaluating all 7 baseline and ablation configurations, verifying held-out seat generalization, and computing the complete acoustic metric suite.

---

## 3. The 3 Best Algorithms

MiniBass incorporates three specialized algorithms implemented in [`minibass/algorithms.py`](file:///home/andromeda/Documents/code/projects/spok/minibass/algorithms.py):

### Algorithm 1: Vectorized Geometric Tensor Scanner (`VectorizedGeometricScanner`)
- **Methodology:** Precomputes complex acoustic pressure tensors $A(a, r, k) = H(r, \operatorname{pos}(a), k) q_a(k)$ for all candidate enclosure-driver-DSP configurations. Uses chunked SIMD broadcasting over 4-box combinations.
- **Throughput:** Evaluates $>50,000$ combinations/second.
- **Objective:** Maximizes tolerance corridor cell pass rate while minimizing room-wide RMS flatness error:
  $$\epsilon_{\text{RMS}} = \sqrt{\frac{1}{R K} \sum_{r=1}^R \sum_{k=1}^K \left( 20 \log_{10} \frac{|p(r, k)|}{p_{\text{target}}} \right)^2}$$

### Algorithm 2: Multi-Seat Minimax Optimizer (`MultiSeatMinimaxOptimizer`)
- **Methodology:** Rather than optimizing only pooled audience averages, this optimizer focuses on audience equality by solving:
  $$\max_{C} \min_{s} \sum_{k=1}^K \mathbb{I}(p_{\text{lo}} \le |p(s, k)| \le p_{\text{hi}})$$
- **Conjunction Spectrum:** Computes $N(k) = |\{s : \text{tones}_s \ge k\}|$ to identify the number of audience seats simultaneously passing at least $k$ frequencies.

### Algorithm 3: Active Bilinear Boundary Dipole Solver (`RLTActiveBilinearSolver`)
- **Methodology:** Reformulation-Linearization Technique (RLT) solver for active boundary absorption / dipole synthesis. Decomposes acoustic sound field power into direct and bilinear cross-terms:
  $$|P_{s, f}|^2 = \sum_i |p_{i, s, f}|^2 + 2 \sum_{i < j} \operatorname{Re}(p_{i, s, f} p_{j, s, f}^*)$$
- **Application:** Discovers phase-inverted boundary cancellation pairs that absorb longitudinal room reflections with minimal physical cabinet displacement.

---

## 4. Benchmark Performance Summary

| Configuration | RMS Error (dB)<br>In-Sample / Held-Out | Worst Seat (dB)<br>In-Sample / Held-Out | Pass Rate ($\pm 3\text{ dB}$)<br>In-Sample / Held-Out | Power (W)<br>Avg / Total |
| :--- | :---: | :---: | :---: | :---: |
| **Single Sub (Corner, Pos 0)** | $9.33\text{ dB}$ / $9.39\text{ dB}$ | $16.11\text{ dB}$ / $14.25\text{ dB}$ | $28.4\%$ / $28.0\%$ | $109.4\text{ W}$ / $10.4\text{ kW}$ |
| **Single Sub (Center, Pos 3)** | $9.62\text{ dB}$ / $9.39\text{ dB}$ | $15.61\text{ dB}$ / $14.97\text{ dB}$ | $25.9\%$ / $24.7\%$ | $109.4\text{ W}$ / $10.4\text{ kW}$ |
| **Symmetric 4-Sub (No DSP)** | $6.99\text{ dB}$ / $7.27\text{ dB}$ | $11.52\text{ dB}$ / $9.61\text{ dB}$ | $38.9\%$ / $36.1\%$ | $194.5\text{ W}$ / $18.5\text{ kW}$ |
| **Ablation 1: Champ Layout (No DSP)** | $6.59\text{ dB}$ / $6.44\text{ dB}$ | $10.84\text{ dB}$ / $9.68\text{ dB}$ | $41.1\%$ / $39.3\%$ | $237.7\text{ W}$ / $22.6\text{ kW}$ |
| **Ablation 2: Champ + Delay Only** | $6.28\text{ dB}$ / $6.19\text{ dB}$ | $9.82\text{ dB}$ / $8.62\text{ dB}$ | $38.9\%$ / $36.0\%$ | $337.9\text{ W}$ / $32.1\text{ kW}$ |
| **Ablation 3: Champ + Lowpass Only** | $6.45\text{ dB}$ / $6.27\text{ dB}$ | $10.13\text{ dB}$ / $8.83\text{ dB}$ | $38.9\%$ / $37.1\%$ | $298.6\text{ W}$ / $28.4\text{ kW}$ |
| **Champion (Asymm + Delay + LP)** | **$6.25\text{ dB}$** / **$6.16\text{ dB}$** | **$9.97\text{ dB}$** / **$9.80\text{ dB}$** | **$39.3\%$** / **$37.6\%$** | **$3.14\text{ W}$** / **$298.6\text{ W}$** |

---

## 5. Champion Array Hardware & DSP Configuration

The synthesized champion array deploys 4 front-stage subwoofers with asymmetric placement and differentiated active DSP:
1. **Subwoofer 1 (Position 2, $x = 3.39\text{ m}$):** 10-inch ported (`box-1458`, $41.5\text{ L}$ gross, $29.8\text{ L}$ net), driver `s03`, Gain: 3.0, Delay: $2.0\text{ ms}$, Filter: Flat.
2. **Subwoofer 2 (Position 3, $x = 4.46\text{ m}$):** 10-inch ported (`box-1458`, driver `s03`), Gain: 3.0, Delay: $2.0\text{ ms}$, Filter: Flat.
3. **Subwoofer 3 (Position 6, $x = 7.68\text{ m}$):** 10-inch ported (`box-1458`, driver `s03`), Gain: 3.0, Delay: $0.0\text{ ms}$, Filter: $120\text{ Hz}$ Lowpass.
4. **Subwoofer 4 (Position 7, $x = 8.75\text{ m}$):** 10-inch ported (`box-83a0`, driver `s04`), Gain: 1.0, Delay: $0.0\text{ ms}$, Filter: $120\text{ Hz}$ Lowpass.

---

## 6. Generated Figures & Artifacts

All high-resolution figures are located in [`docs/figures/`](file:///home/andromeda/Documents/code/projects/spok/docs/figures):
- `dryrun_10x12_one_sheet_composite.png` (and `.svg`): Comprehensive overview composite.
- `dryrun_10x12_baselines_ablations.png` (and `.svg`): 4-panel comparison of in-sample vs. held-out RMS error, minimax worst-seat deviation, cell pass rate, and Toole spatial standard deviation.
- `dryrun_10x12_physical_monte_carlo.png` (and `.svg`): Histograms of the 256 physical uncertainty realizations.
- `dryrun_10x12_room_plan_acoustic_error.png` (and `.svg`): 2D room layout with stage, seats, active subwoofers, and continuous modal error field.
- `dryrun_10x12_frequency_response_corridor.png` (and `.svg`): All 80 seat curves, spatial mean, and $\pm 1$ SD corridor.
- `dryrun_10x12_modal_slices_20_50_100_200hz.png` (and `.svg`): Exact Helmholtz Green's function standing wave slices at 20, 50, 100, and 200 Hz.
- `dryrun_10x12_seat_histogram.png` (and `.svg`): Passing tone distribution across audience seats.

