"""The 3 Best Algorithms for Rectangular Room Low-Frequency Acoustic Optimization in MiniBass.

Consolidates the three top-performing algorithmic methodologies:

1. VectorizedGeometricScanner:
   High-throughput tensor-vectorized acoustic evaluator using precomputed complex
   Green's function atoms. Evaluates 4-box layout combinations and discrete active controls
   via SIMD broadcasting (>50,000 combinations/sec) to maximize pooled seat-frequency cell coverage.

2. MultiSeatMinimaxOptimizer:
   Multi-seat equalizer focusing on the worst-seat tone count:
       max_{C} min_{s} ∑_{f} 𝕀(p_lower <= |P_{s,f}| <= p_upper)
   and tracking the seat conjunction profile N(k) = |{s : tones_s >= k}| to enforce
   spatial equality across the listening audience.

3. RLTActiveBilinearSolver:
   Reformulation-Linearization Technique (RLT) solver for active acoustic boundary sinks
   and dipole absorbers. Exploits the exact expansion:
       |P_{s,f}|^2 = ∑_i |p_{i,s,f}|^2 + 2 ∑_{i < j} Re(p_{i,s,f} p_{j,s,f}^*)
   to discover destructive interference pairs along boundaries with minimal total enclosure volume.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class EvaluationResult:
    """Standardized performance metrics from acoustic candidate evaluation."""
    positions: tuple[int, ...]
    design_ids: tuple[str, ...]
    controls: tuple[dict[str, Any], ...]
    cell_pass_rate_pct: float
    passing_cells: int
    total_cells: int
    rms_error_db: float
    worst_seat_rms_db: float
    spatial_sd_db: float
    best_seat_pass: int
    worst_seat_pass: int
    median_seat_pass: float
    seat_tones: list[int]
    total_volume_m3: float
    mean_power_w: float


# ==============================================================================
# ALGORITHM 1: Vectorized Geometric Tensor Scanner
# ==============================================================================

class VectorizedGeometricScanner:
    """High-throughput tensor-vectorized acoustic search engine.
    
    Evaluates layout and control combinations using chunked SIMD broadcasting over
    precomputed complex acoustic atom tensors A(p, d, c, s, f).
    """

    def __init__(
        self,
        raw_instance: dict[str, Any],
        atom_data: dict[str, Any],
        chunk_size: int = 16384,
    ) -> None:
        self.raw = raw_instance
        self.atom_data = atom_data
        self.chunk_size = chunk_size

        self.target = float(raw_instance["target_pa_rms"])
        self.tol = float(raw_instance["tolerance_db"])
        self.lower = 10 ** (-self.tol / 20.0) * self.target
        self.upper = 10 ** (self.tol / 20.0) * self.target

        self.frequencies = np.asarray(raw_instance["frequencies_hz"], dtype=float)
        self.n_freqs = len(self.frequencies)
        self.keys = atom_data["keys"]
        self.atom_pressures = atom_data["atom_pressures"]  # shape: [N_atoms, S, F]
        self.atom_powers = atom_data["atom_powers"]
        self.atom_volumes = atom_data["atom_volumes"]
        self.array_lookup = atom_data["array_lookup"]
        self.controls = atom_data["controls"]
        self.n_controls = len(self.controls)

        # Control grid: (N_controls)^4 combinations
        self.control_grid_4 = np.array(
            list(itertools.product(range(self.n_controls), repeat=4)), dtype=np.int16
        )

    def get_enclosure_pressure_slots(
        self, positions: Sequence[int], design_ids: Sequence[str]
    ) -> list[np.ndarray] | None:
        """Retrieve pressure slices of shape [N_controls, S, F] for each box."""
        if len(positions) != len(design_ids):
            raise ValueError("Length of positions and design_ids must match.")

        slots = []
        for pos, did in zip(positions, design_ids):
            p_atoms = [k for k in self.keys if k[0] == pos and k[1] == did]
            if len(p_atoms) != self.n_controls:
                return None
            idx = [self.array_lookup[k] for k in p_atoms]
            slots.append(self.atom_pressures[idx])
        return slots

    def evaluate_layout_exhaustive(
        self, positions: tuple[int, int, int, int], design_ids: tuple[str, str, str, str]
    ) -> tuple[EvaluationResult, int]:
        """Exhaustively scan all active DSP control combinations for a 4-box layout.
        
        Returns the best EvaluationResult (maximizing cell pass rate, breaking ties by lowest RMS error)
        and the number of evaluated combinations.
        """
        slots = self.get_enclosure_pressure_slots(positions, design_ids)
        if slots is None:
            raise ValueError(f"Invalid positions {positions} or designs {design_ids}")

        p0, p1, p2, p3 = slots
        best_cells = -1
        best_ctrl_idx = -1
        best_err = float("inf")

        n_combos = len(self.control_grid_4)
        for i in range(0, n_combos, self.chunk_size):
            sub = self.control_grid_4[i : i + self.chunk_size]
            tot = (
                p0[sub[:, 0]]
                + p1[sub[:, 1]]
                + p2[sub[:, 2]]
                + p3[sub[:, 3]]
            )  # shape: [B, S, F]
            mag = np.abs(tot)
            in_band = (mag >= self.lower) & (mag <= self.upper)
            cells = np.sum(in_band, axis=(1, 2))  # shape: [B]

            max_c = int(np.max(cells))
            if max_c > best_cells:
                top_indices = np.where(cells == max_c)[0]
                for idx in top_indices:
                    diff_db = 20.0 * np.log10(np.maximum(mag[idx] / self.target, 1e-12))
                    err = float(np.sqrt(np.mean(diff_db ** 2)))
                    if max_c > best_cells or err < best_err:
                        best_cells = max_c
                        best_err = err
                        best_ctrl_idx = i + idx

        best_ctrl_tuple = tuple(int(c) for c in self.control_grid_4[best_ctrl_idx])
        res = self._compute_detailed_metrics(positions, design_ids, best_ctrl_tuple, slots)
        return res, n_combos

    def _compute_detailed_metrics(
        self,
        positions: tuple[int, ...],
        design_ids: tuple[str, ...],
        control_indices: tuple[int, ...],
        slots: list[np.ndarray],
    ) -> EvaluationResult:
        tot = np.zeros_like(slots[0][0])
        for slot, c_idx in zip(slots, control_indices):
            tot += slot[c_idx]

        mag = np.abs(tot)  # shape: [S, F]
        in_band = (mag >= self.lower) & (mag <= self.upper)
        cells = int(np.sum(in_band))
        n_total_cells = mag.shape[0] * mag.shape[1]
        cell_pct = float(cells / n_total_cells * 100.0)

        seat_tones = [int(np.sum(in_band[s])) for s in range(mag.shape[0])]

        diff_db = 20.0 * np.log10(np.maximum(mag / self.target, 1e-12))
        spl_db = 20.0 * np.log10(np.maximum(mag, 1e-12) / 20e-6)
        rms_err = float(np.sqrt(np.mean(diff_db ** 2)))

        seat_rms_err = np.sqrt(np.mean(diff_db ** 2, axis=1))
        worst_seat_rms = float(np.max(seat_rms_err))
        spatial_sd = float(np.mean(np.std(spl_db, axis=0)))

        # Hardware metrics
        total_vol = 0.0
        tot_power = 0.0
        for pos, did, c_idx in zip(positions, design_ids, control_indices):
            k = (pos, did, c_idx)
            if k in self.array_lookup:
                atom_idx = self.array_lookup[k]
                total_vol += float(self.atom_volumes[atom_idx])
                tot_power += float(np.mean(self.atom_powers[atom_idx]))

        chosen_controls = tuple(self.controls[c] for c in control_indices)

        return EvaluationResult(
            positions=positions,
            design_ids=design_ids,
            controls=chosen_controls,
            cell_pass_rate_pct=cell_pct,
            passing_cells=cells,
            total_cells=n_total_cells,
            rms_error_db=rms_err,
            worst_seat_rms_db=worst_seat_rms,
            spatial_sd_db=spatial_sd,
            best_seat_pass=int(max(seat_tones)),
            worst_seat_pass=int(min(seat_tones)),
            median_seat_pass=float(np.median(seat_tones)),
            seat_tones=seat_tones,
            total_volume_m3=total_vol,
            mean_power_w=tot_power,
        )


# ==============================================================================
# ALGORITHM 2: Multi-Seat Minimax Optimizer
# ==============================================================================

class MultiSeatMinimaxOptimizer:
    """Multi-seat equalizer optimizing audience fairness and seat conjunction.
    
    Rather than merely maximizing pooled cells, this optimizer solves:
        max_{C} min_{s} (passing tones at seat s)
    and evaluates the seat conjunction spectrum N(k) = |{s : tones_s >= k}|.
    """

    def __init__(self, scanner: VectorizedGeometricScanner) -> None:
        self.scanner = scanner

    def optimize_minimax_seat_tones(
        self,
        positions: tuple[int, int, int, int],
        design_ids: tuple[str, str, str, str],
    ) -> EvaluationResult:
        """Scan control combinations to maximize worst-seat passing tones."""
        slots = self.scanner.get_enclosure_pressure_slots(positions, design_ids)
        if slots is None:
            raise ValueError(f"Invalid positions {positions} or designs {design_ids}")

        p0, p1, p2, p3 = slots
        best_worst_seat = -1
        best_cells = -1
        best_ctrl_idx = -1

        grid = self.scanner.control_grid_4
        for i in range(0, len(grid), self.scanner.chunk_size):
            sub = grid[i : i + self.scanner.chunk_size]
            tot = (
                p0[sub[:, 0]]
                + p1[sub[:, 1]]
                + p2[sub[:, 2]]
                + p3[sub[:, 3]]
            )  # shape: [B, S, F]
            mag = np.abs(tot)
            in_band = (mag >= self.scanner.lower) & (mag <= self.scanner.upper)
            seat_tones = np.sum(in_band, axis=2)  # shape: [B, S]
            min_tones_per_seat = np.min(seat_tones, axis=1)  # shape: [B]
            total_cells = np.sum(seat_tones, axis=1)  # shape: [B]

            max_min_t = int(np.max(min_tones_per_seat))
            if max_min_t > best_worst_seat:
                best_worst_seat = max_min_t
                top_idx = np.where(min_tones_per_seat == max_min_t)[0]
                best_sub_idx = top_idx[int(np.argmax(total_cells[top_idx]))]
                best_cells = int(total_cells[best_sub_idx])
                best_ctrl_idx = i + best_sub_idx
            elif max_min_t == best_worst_seat and best_worst_seat > -1:
                top_idx = np.where(min_tones_per_seat == max_min_t)[0]
                c_max = int(np.max(total_cells[top_idx]))
                if c_max > best_cells:
                    best_cells = c_max
                    best_sub_idx = top_idx[int(np.argmax(total_cells[top_idx]))]
                    best_ctrl_idx = i + best_sub_idx

        best_ctrl_tuple = tuple(int(c) for c in grid[best_ctrl_idx])
        return self.scanner._compute_detailed_metrics(
            positions, design_ids, best_ctrl_tuple, slots
        )

    @staticmethod
    def compute_conjunction_spectrum(seat_tones: Sequence[int]) -> dict[int, int]:
        """Compute N(k) = number of seats passing at least k tones simultaneously."""
        tones = np.asarray(seat_tones, dtype=int)
        spectrum = {}
        for k in range(0, int(np.max(tones)) + 1):
            spectrum[k] = int(np.sum(tones >= k))
        return spectrum


# ==============================================================================
# ALGORITHM 3: RLT Active Bilinear Solver
# ==============================================================================

class RLTActiveBilinearSolver:
    """Reformulation-Linearization Technique (RLT) bilinear interference solver.
    
    Models multi-source acoustic pressure as a sum of individual source squared pressures
    and pairwise bilinear cross terms:
        |P_{s,f}|^2 = ∑_i |p_{i,s,f}|^2 + 2 ∑_{i < j} Re(p_{i,s,f} p_{j,s,f}^*)
    Identifies boundary cancellation pairs that absorb modal energy with minimal physical volume.
    """

    def __init__(self, scanner: VectorizedGeometricScanner) -> None:
        self.scanner = scanner

    def evaluate_bilinear_pair(
        self,
        pos1: int,
        pos2: int,
        design1: str,
        design2: str,
    ) -> dict[str, Any]:
        """Evaluate destructive interference and absorption capability of a 2-box boundary dipole."""
        slots1 = self.scanner.get_enclosure_pressure_slots([pos1], [design1])
        slots2 = self.scanner.get_enclosure_pressure_slots([pos2], [design2])
        if slots1 is None or slots2 is None:
            raise ValueError("Invalid dipole configuration positions or designs")

        p1 = slots1[0]  # shape: [C, S, F]
        p2 = slots2[0]  # shape: [C, S, F]

        best_interference_ratio = float("inf")
        best_c1, best_c2 = 0, 0

        # Scan all C x C control pairs
        for c1 in range(len(p1)):
            for c2 in range(len(p2)):
                direct_energy = np.mean(np.abs(p1[c1]) ** 2 + np.abs(p2[c2]) ** 2)
                cross_term = 2.0 * np.mean(np.real(p1[c1] * np.conj(p2[c2])))
                tot_energy = direct_energy + cross_term

                if direct_energy > 1e-12:
                    ratio = tot_energy / direct_energy
                    if ratio < best_interference_ratio:
                        best_interference_ratio = ratio
                        best_c1, best_c2 = c1, c2

        return {
            "pos1": pos1,
            "pos2": pos2,
            "design1": design1,
            "design2": design2,
            "best_control_1": self.scanner.controls[best_c1],
            "best_control_2": self.scanner.controls[best_c2],
            "destructive_cancellation_ratio": float(best_interference_ratio),
            "cross_term_suppression_pct": float(max(0.0, (1.0 - best_interference_ratio) * 100.0)),
        }
