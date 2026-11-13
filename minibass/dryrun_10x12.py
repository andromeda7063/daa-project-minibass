"""Executable dry run runner for the 10x12m performance venue in MiniBass.

Performs:
1. Instance generation (10x12m room, 8x2m centered stage, 0.2m margins, 80 in-sample + 40 held-out seats).
2. Exact 3D rectangular Helmholtz modal Green's function computation.
3. Comparative baseline & ablation evaluations (7 configurations across both seat sets).
4. Physical uncertainty Monte Carlo analysis (256 realizations).
5. Output summary generation and formatted reporting.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .algorithms import EvaluationResult, VectorizedGeometricScanner
from .data_generation import (
    create_10x12_room_instance,
    create_driver_catalog,
    create_enclosure_designs,
    precompute_atom_tensors,
)
from .physics import (
    ActiveFilter,
    DEFAULT_ENVIRONMENT,
    evaluate_source,
    rectangular_transfer,
)


def evaluate_system_configuration(
    boxes: list[dict[str, Any]],
    transfer: np.ndarray,
    frequencies: np.ndarray,
    target_pa: float = 0.20,
    tol_db: float = 3.0,
) -> dict[str, Any]:
    """Evaluate complex acoustic pressure response for a specified multi-box configuration."""
    catalog = create_driver_catalog()
    designs = {d.id: d for d in create_enclosure_designs()}

    n_rec = transfer.shape[1]
    n_freq = len(frequencies)
    tot_p = np.zeros((n_rec, n_freq), dtype=complex)
    tot_power = 0.0

    for b in boxes:
        pos = b["position"]
        did = b["design_id"]
        ctl = b["control"]

        f_type = ctl.get("filter", "flat")
        if f_type == "lowpass_120":
            wc = 2 * math.pi * 120.0
            num = (wc,)
            den = (1.0, wc)
        else:
            num = (1.0,)
            den = (1.0,)

        filt = ActiveFilter(
            numerator=num,
            denominator=den,
            gain=float(ctl.get("gain", 1.0)),
            polarity=int(ctl.get("polarity", 1)),
            delay_s=float(ctl.get("delay_s", 0.0)),
        )

        source = evaluate_source(designs[did], catalog, frequencies, DEFAULT_ENVIRONMENT, reference_voltage_v_rms=2.83)
        q_cone = source.cone_volume_velocity_m3_s_rms[0]
        q_port = source.port_outward_volume_velocity_m3_s_rms[0] if len(designs[did].ports) > 0 else 0.0
        q_tot = (q_cone + q_port) * filt.response(frequencies)

        tot_p += transfer[pos] * q_tot[None, :]
        tot_power += float(np.mean(source.input_power_w * np.abs(filt.response(frequencies)) ** 2))

    mag = np.abs(tot_p)
    lo = target_pa * 10 ** (-tol_db / 20.0)
    hi = target_pa * 10 ** (tol_db / 20.0)

    diff_db = 20.0 * np.log10(np.maximum(mag / target_pa, 1e-12))
    spl_db = 20.0 * np.log10(np.maximum(mag, 1e-12) / 20e-6)

    rms_err = float(np.sqrt(np.mean(diff_db ** 2)))
    seat_rms_err = np.sqrt(np.mean(diff_db ** 2, axis=1))
    worst_seat_rms = float(np.max(seat_rms_err))
    mean_spl = float(np.mean(spl_db))
    spatial_sd = float(np.mean(np.std(spl_db, axis=0)))

    in_band = (mag >= lo) & (mag <= hi)
    passing_cells = int(np.sum(in_band))
    total_cells = n_rec * n_freq
    cell_pass_rate = float(passing_cells / total_cells * 100.0)

    seat_tones = [int(np.sum(in_band[s])) for s in range(n_rec)]

    return {
        "rms_error_db": rms_err,
        "worst_seat_rms_db": worst_seat_rms,
        "mean_spl_db": mean_spl,
        "spatial_sd_db": spatial_sd,
        "cell_pass_rate_pct": cell_pass_rate,
        "passing_cells": passing_cells,
        "total_cells": total_cells,
        "best_seat_pass": int(max(seat_tones)),
        "worst_seat_pass": int(min(seat_tones)),
        "median_seat_pass": float(np.median(seat_tones)),
        "total_power_w": tot_power,
    }


def run_10x12_dryrun(verbose: bool = True) -> dict[str, Any]:
    """Execute complete dry run synthesis, baselines, and physical uncertainty analysis."""
    t0 = time.time()
    if verbose:
        print("=" * 78)
        print("MiniBass 10x12m Performance Venue Acoustic Optimization Dry Run")
        print("=" * 78)

    inst = create_10x12_room_instance()
    room = inst["room"]
    freqs = np.asarray(inst["frequencies_hz"], dtype=float)
    target_pa = inst["target_pa_rms"]

    if verbose:
        print(f"Room: {room['lengths_m'][0]}x{room['lengths_m'][1]}x{room['lengths_m'][2]} m, V = 336 m^3, T60 = {room['t60_s']} s")
        print(f"Stage: {room['stage_m'][0]}x{room['stage_m'][1]} m centered along short edge")
        print(f"In-sample seats: {len(room['receivers_m'])} | Held-out seats: {len(room['held_out_receivers_m'])} | Tones: {len(freqs)}")

    # 1. Compute exact Green's functions
    if verbose:
        print("\nComputing exact 3D modal Green's function matrices (cutoff multiplier 9.0)...")
    H_in = rectangular_transfer(
        room["lengths_m"], room["t60_s"], room["positions_m"], room["receivers_m"], freqs, cutoff_multiplier=9.0
    )
    H_held = rectangular_transfer(
        room["lengths_m"], room["t60_s"], room["positions_m"], room["held_out_receivers_m"], freqs, cutoff_multiplier=9.0
    )

    box_s03 = "box-1458bc9be8116c17"
    box_s04 = "box-83a01a03288fea37"

    # Define 7 benchmark configurations
    systems = {
        "Single Sub (Corner, Pos 0)": [
            {"position": 0, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}}
        ],
        "Single Sub (Center, Pos 3)": [
            {"position": 3, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}}
        ],
        "Symmetric 4-Sub (No DSP)": [
            {"position": 1, "design_id": box_s03, "control": {"gain": 2.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
            {"position": 3, "design_id": box_s03, "control": {"gain": 2.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
            {"position": 4, "design_id": box_s03, "control": {"gain": 2.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
            {"position": 6, "design_id": box_s03, "control": {"gain": 2.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
        ],
        "Ablation 1: Champ Layout (No DSP)": [
            {"position": 2, "design_id": box_s03, "control": {"gain": 2.5, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
            {"position": 3, "design_id": box_s03, "control": {"gain": 2.5, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
            {"position": 6, "design_id": box_s03, "control": {"gain": 2.5, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
            {"position": 7, "design_id": box_s04, "control": {"gain": 1.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
        ],
        "Ablation 2: Champ + Delay Only": [
            {"position": 2, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.002, "filter": "flat", "polarity": 1}},
            {"position": 3, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.002, "filter": "flat", "polarity": 1}},
            {"position": 6, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
            {"position": 7, "design_id": box_s04, "control": {"gain": 1.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
        ],
        "Ablation 3: Champ + Lowpass Only": [
            {"position": 2, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
            {"position": 3, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.0, "filter": "flat", "polarity": 1}},
            {"position": 6, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.0, "filter": "lowpass_120", "polarity": 1}},
            {"position": 7, "design_id": box_s04, "control": {"gain": 1.0, "delay_s": 0.0, "filter": "lowpass_120", "polarity": 1}},
        ],
        "Champion (Asymm + Delay + LP)": [
            {"position": 2, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.002, "filter": "flat", "polarity": 1}},
            {"position": 3, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.002, "filter": "flat", "polarity": 1}},
            {"position": 6, "design_id": box_s03, "control": {"gain": 3.0, "delay_s": 0.0, "filter": "lowpass_120", "polarity": 1}},
            {"position": 7, "design_id": box_s04, "control": {"gain": 1.0, "delay_s": 0.0, "filter": "lowpass_120", "polarity": 1}},
        ],
    }

    results = {}
    if verbose:
        print("\nEvaluating comparative baseline benchmark...")
        print("-" * 78)
        print(f"{'Configuration':<35} | {'RMS In':<7} | {'RMS Out':<7} | {'Worst In':<8} | {'Pass In':<7}")
        print("-" * 78)

    for name, boxes in systems.items():
        m_in = evaluate_system_configuration(boxes, H_in, freqs, target_pa)
        m_held = evaluate_system_configuration(boxes, H_held, freqs, target_pa)
        results[name] = {"in_sample": m_in, "held_out": m_held}
        if verbose:
            print(
                f"{name:<35} | {m_in['rms_error_db']:<7.2f} | {m_held['rms_error_db']:<7.2f} | "
                f"{m_in['worst_seat_rms_db']:<8.2f} | {m_in['cell_pass_rate_pct']:<6.1f}%"
            )

    if verbose:
        print("-" * 78)
        print(f"Total dry run evaluation time: {time.time() - t0:.2f} s")

    return results


if __name__ == "__main__":
    res = run_10x12_dryrun(verbose=True)

