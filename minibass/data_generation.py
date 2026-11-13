"""Data generation for room acoustic venues, transducer catalogs, and atom tensors in MiniBass.

Provides:
- 10x12m performance venue configuration with centered stage and safety margins
- In-sample (80 seats) and held-out (40 off-grid seats) receiver coordinates
- Discrete Thiele-Small driver and enclosure catalog
- Active DSP control configurations
- Atom tensor precomputation for SIMD vectorized optimization
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .physics import (
    ActiveFilter,
    DEFAULT_ENVIRONMENT,
    Design,
    Driver,
    DriverMount,
    Port,
    evaluate_source,
    net_volume_m3,
)


def create_10x12_room_instance() -> dict[str, Any]:
    """Generate the user-specified 10x12m venue with centered 8x2m stage and 0.2m margins."""
    lengths = [10.0, 12.0, 2.8]
    t60 = 0.45
    target_pa = 0.20  # 80.0 dB SPL
    tol_db = 3.0

    # 8 candidate subwoofer locations along the front stage edge (y = 1.78m, z = 0.40m)
    positions = [[float(x), 1.78, 0.40] for x in np.linspace(1.25, 8.75, 8)]

    # Audience seating zone: exactly 0.2m margins from stage front (y=2.0 -> y>=2.2) and walls
    # Width: x in [0.2, 9.8] m (9.6m wide); Depth: y in [2.2, 11.8] m (9.6m deep)
    # 80 In-sample seats: 10 columns x 8 rows, cell-centered at ear height z = 1.10m
    xs_in = np.linspace(0.2 + (9.6 / 10) / 2, 9.8 - (9.6 / 10) / 2, 10)
    ys_in = np.linspace(2.2 + (9.6 / 8) / 2, 11.8 - (9.6 / 8) / 2, 8)
    receivers_in = [[float(x), float(y), 1.10] for y in ys_in for x in xs_in]

    # 40 Held-out seats: 8 columns x 5 rows shifted off-grid to audit continuous generalization
    xs_held = np.linspace(0.68, 9.32, 8)
    ys_held = np.linspace(2.80, 11.20, 5)
    receivers_held = [[float(x), float(y), 1.10] for y in ys_held for x in xs_held]

    # Refined 95 discrete frequencies from 20 to 200 Hz
    frequencies = np.geomspace(20.0, 200.0, 95).tolist()

    return {
        "name": "minibass_10x12_performance_venue",
        "room": {
            "lengths_m": lengths,
            "t60_s": t60,
            "stage_m": [8.0, 2.0],
            "positions_m": positions,
            "receivers_m": receivers_in,
            "held_out_receivers_m": receivers_held,
        },
        "target_pa_rms": target_pa,
        "tolerance_db": tol_db,
        "frequencies_hz": frequencies,
        "limits": {
            "max_boxes": 4,
            "beam_width": 32,
            "cutoff_multiplier": 9.0,
        },
    }


def create_driver_catalog() -> dict[str, Driver]:
    """Discrete menu of high-compliance 10-inch and 12-inch transducers."""
    return {
        "s01": Driver(
            id="s01",
            re_ohm=3.2,
            le_h=0.0012,
            bl_n_a=14.2,
            moving_mass_kg=0.095,
            compliance_m_n=0.00038,
            resistance_n_s_m=2.8,
            piston_area_m2=0.035,
            diameter_m=0.26,
            depth_m=0.14,
            displaced_volume_m3=0.0022,
        ),
        "s02": Driver(
            id="s02",
            re_ohm=3.4,
            le_h=0.0014,
            bl_n_a=15.8,
            moving_mass_kg=0.120,
            compliance_m_n=0.00032,
            resistance_n_s_m=3.2,
            piston_area_m2=0.035,
            diameter_m=0.26,
            depth_m=0.15,
            displaced_volume_m3=0.0025,
        ),
        "s03": Driver(
            id="s03",
            re_ohm=3.1,
            le_h=0.0011,
            bl_n_a=15.0,
            moving_mass_kg=0.088,
            compliance_m_n=0.00042,
            resistance_n_s_m=2.5,
            piston_area_m2=0.035,
            diameter_m=0.26,
            depth_m=0.14,
            displaced_volume_m3=0.0021,
        ),
        "s04": Driver(
            id="s04",
            re_ohm=3.6,
            le_h=0.0015,
            bl_n_a=16.5,
            moving_mass_kg=0.135,
            compliance_m_n=0.00028,
            resistance_n_s_m=3.5,
            piston_area_m2=0.035,
            diameter_m=0.26,
            depth_m=0.15,
            displaced_volume_m3=0.0026,
        ),
    }


def create_enclosure_designs() -> tuple[Design, ...]:
    """Discrete catalog of ported and sealed subwoofer enclosures."""
    # box-1458bc9be8116c17: 10" ported cabinet (0.36 x 0.36 x 0.32m, 41.5L gross, 29.8L net)
    port_10 = Port(
        centre_m=(0.26, 0.26),
        inner_radius_m=0.035,
        outer_radius_m=0.040,
        length_m=0.22,
        effective_length_m=0.26,
        resistance_pa_s_m3=180.0,
    )
    mount_s03 = DriverMount(driver_id="s03", centre_m=(0.18, 0.18))
    mount_s04 = DriverMount(driver_id="s04", centre_m=(0.18, 0.18))

    box_s03 = Design(
        id="box-1458bc9be8116c17",
        outer_dimensions_m=(0.36, 0.36, 0.32),
        wall_thickness_m=0.018,
        mounts=(mount_s03,),
        ports=(port_10,),
    )
    box_s04 = Design(
        id="box-83a01a03288fea37",
        outer_dimensions_m=(0.36, 0.36, 0.32),
        wall_thickness_m=0.018,
        mounts=(mount_s04,),
        ports=(port_10,),
    )
    return (box_s03, box_s04)


def create_control_records() -> list[dict[str, Any]]:
    """Generate the 16 active DSP control states."""
    controls = []
    gains = [1.0, 3.0]
    delays = [0.0, 0.002]
    filters = ["flat", "lowpass_120"]
    polarities = [1, -1]

    for g in gains:
        for d in delays:
            for f in filters:
                for pol in [1]:  # default non-inverted polarity states
                    controls.append({"gain": g, "delay_s": d, "filter": f, "polarity": pol})

    # Pad to 16 with dipole inversion options
    for g in gains:
        for d in delays:
            controls.append({"gain": g, "delay_s": d, "filter": "flat", "polarity": -1})

    return controls[:16]


def active_filter_from_record(record: dict[str, Any]) -> ActiveFilter:
    """Build ActiveFilter instance from control dictionary."""
    f_type = record.get("filter", "flat")
    if f_type == "lowpass_120":
        wc = 2 * math.pi * 120.0
        num = (wc,)
        den = (1.0, wc)
    else:
        num = (1.0,)
        den = (1.0,)
    return ActiveFilter(
        numerator=num,
        denominator=den,
        gain=float(record.get("gain", 1.0)),
        polarity=int(record.get("polarity", 1)),
        delay_s=float(record.get("delay_s", 0.0)),
    )


def precompute_atom_tensors(
    raw_instance: dict[str, Any],
    transfer: np.ndarray,
    frequencies: Sequence[float],
) -> dict[str, Any]:
    """Precompute complex acoustic atom tensors A(a, r, k) for fast SIMD search.
    
    Shape of atom_pressure: [num_atoms, num_receivers, num_frequencies]
    """
    catalog = create_driver_catalog()
    designs = create_enclosure_designs()
    controls = create_control_records()

    n_pos = len(raw_instance["room"]["positions_m"])
    n_rec = len(raw_instance["room"]["receivers_m"])
    n_freq = len(frequencies)

    atom_pressures = []
    atom_powers = []
    atom_volumes = []
    keys = []
    array_lookup = {}

    for pos in range(n_pos):
        for des in designs:
            source = evaluate_source(des, catalog, frequencies, DEFAULT_ENVIRONMENT, reference_voltage_v_rms=2.83)
            cone_q = source.cone_volume_velocity_m3_s_rms[0]  # shape: [F]
            port_q = source.port_outward_volume_velocity_m3_s_rms[0] if len(des.ports) > 0 else 0.0
            total_flow = cone_q + port_q  # shape: [F]
            vol = math.prod(des.outer_dimensions_m)

            for ci, c_rec in enumerate(controls):
                ctl = active_filter_from_record(c_rec)
                resp = ctl.response(frequencies)  # shape: [F]
                flow_dsp = total_flow * resp
                power_dsp = source.input_power_w * np.abs(resp) ** 2

                # Room transfer for this position: transfer[pos] has shape [R, F]
                p_atom = transfer[pos] * flow_dsp[None, :]  # shape [R, F]

                idx = len(atom_pressures)
                atom_pressures.append(p_atom)
                atom_powers.append(power_dsp)
                atom_volumes.append(vol)

                k = (pos, des.id, ci)
                keys.append(k)
                array_lookup[k] = idx

    return {
        "atom_pressures": np.asarray(atom_pressures, dtype=complex),
        "atom_powers": np.asarray(atom_powers, dtype=float),
        "atom_volumes": np.asarray(atom_volumes, dtype=float),
        "keys": keys,
        "array_lookup": array_lookup,
        "designs": {d.id: d for d in designs},
        "controls": controls,
    }
