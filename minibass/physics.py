"""Physical acoustics and coupled electromechanical transducer models for MiniBass.

Provides:
- Exact 3D rectangular Helmholtz modal Green's function (rectangular_transfer)
- Coupled 4th-order electromechanical Thiele-Small matrix solver (evaluate_source)
- Active DSP filter transfer functions (ActiveFilter)
- Enclosure cavity compliance and net volume calculations
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Sequence

import numpy as np


@dataclass(frozen=True)
class Environment:
    density_kg_m3: float = 1.20
    sound_speed_m_s: float = 343.0


DEFAULT_ENVIRONMENT = Environment()


def _finite(value: Any) -> bool:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive(*values: float) -> bool:
    return all(_finite(v) and v > 0 for v in values)


def _nonnegative(*values: float) -> bool:
    return all(_finite(v) and v >= 0 for v in values)


def rectangular_transfer(
    lengths: Sequence[float],
    t60: float,
    sources: Sequence[Sequence[float]],
    receivers: Sequence[Sequence[float]],
    frequencies: Sequence[float],
    *,
    rho: float = 1.20,
    c: float = 343.0,
    cutoff_multiplier: float = 9.0,
    mode_chunk: int = 4096,
) -> np.ndarray:
    """Exact vectorized 3D rectangular Helmholtz modal Green's function.

    Computes the spatial transfer impedance H(x_s, x_r, omega) in Pa*s/m^3
    using the uncoupled rectangular eigenfunction expansion with exact
    Neumann normalization factors and uniform modal damping delta = 6.91 / t60.
    
    Output shape: [num_sources, num_receivers, num_frequencies]
    """
    L = np.asarray(lengths, float)
    xs = np.asarray(sources, float)
    xr = np.asarray(receivers, float)
    fs = np.asarray(frequencies, float)

    if (
        L.shape != (3,)
        or xs.ndim != 2
        or xs.shape[1] != 3
        or xr.ndim != 2
        or xr.shape[1] != 3
        or fs.ndim != 1
        or len(fs) == 0
        or not all(np.all(np.isfinite(a)) for a in (L, xs, xr, fs))
        or np.any(L <= 0)
        or np.any(xs < 0)
        or np.any(xs > L)
        or np.any(xr < 0)
        or np.any(xr > L)
        or np.any(fs <= 0)
        or not all(math.isfinite(float(x)) and x > 0 for x in (t60, rho, c, cutoff_multiplier))
        or type(mode_chunk) is not int
        or mode_chunk <= 0
    ):
        raise ValueError("invalid rectangular transfer arguments")

    upper = cutoff_multiplier * float(np.max(fs))
    bounds = np.floor(2 * upper * L / c).astype(int)
    triples = np.asarray(
        [
            (a, b, d)
            for a in range(bounds[0] + 1)
            for b in range(bounds[1] + 1)
            for d in range(bounds[2] + 1)
        ],
        dtype=int,
    )
    eigen2 = (np.pi * c) ** 2 * np.sum((triples / L) ** 2, axis=1)
    keep = eigen2 <= (2 * np.pi * upper) ** 2
    triples, eigen2 = triples[keep], eigen2[keep]

    # Modal norm inverse: prod((2/L if n > 0 else 1/L))
    inv_norm = np.prod(np.where(triples == 0, 1.0 / L, 2.0 / L), axis=1)

    omega = 2 * np.pi * fs
    delta = 6.91 / t60
    answer = np.zeros((len(xs), len(xr), len(fs)), dtype=complex)

    for start in range(0, len(triples), mode_chunk):
        stop = min(start + mode_chunk, len(triples))
        chunk_triples = triples[start:stop]
        chunk_source = np.prod(np.cos(np.pi * xs[:, None, :] * chunk_triples[None, :, :] / L), axis=2)
        chunk_receiver = np.prod(np.cos(np.pi * xr[:, None, :] * chunk_triples[None, :, :] / L), axis=2)

        lam = eigen2[start:stop, None]
        denom = lam - omega[None, :] ** 2 + 2j * delta * omega[None, :]
        weights = inv_norm[start:stop, None] * (1j * rho * omega[None, :] * c * c) / denom
        chunk_sw = (chunk_source[:, None, :] * weights.T[None, :, :]).reshape(len(xs) * len(fs), stop - start)
        answer += (chunk_sw @ chunk_receiver.T).reshape(len(xs), len(fs), len(xr)).transpose(0, 2, 1)

    return answer


@dataclass(frozen=True)
class ActiveFilter:
    """Active analog filter with voltage gain, pure delay, polarity, and transfer function."""
    numerator: tuple[float, ...] = (1.0,)
    denominator: tuple[float, ...] = (1.0,)
    gain: float = 1.0
    polarity: int = 1
    delay_s: float = 0.0

    def response(self, frequencies: Sequence[float]) -> np.ndarray:
        f = np.asarray(frequencies, float)
        s = 2j * np.pi * f
        with np.errstate(all="ignore"):
            num = np.polyval(self.numerator, s)
            den = np.polyval(self.denominator, s)
            h = num / den
        out = self.gain * self.polarity * np.exp(-s * self.delay_s) * h
        return np.where(np.isfinite(out), out, 0.0)


@dataclass(frozen=True)
class Driver:
    """Electrodynamic moving-coil transducer Thiele-Small lumped parameters."""
    id: str
    re_ohm: float
    le_h: float
    bl_n_a: float
    moving_mass_kg: float
    compliance_m_n: float
    resistance_n_s_m: float
    piston_area_m2: float
    diameter_m: float
    depth_m: float
    displaced_volume_m3: float
    valid_band_hz: tuple[float, float] = (15.0, 300.0)


@dataclass(frozen=True)
class Port:
    """Acoustic Helmholtz resonator port duct parameters."""
    centre_m: tuple[float, float]
    inner_radius_m: float
    outer_radius_m: float
    length_m: float
    effective_length_m: float
    resistance_pa_s_m3: float

    def impedance(self, frequencies: Sequence[float], env: Environment = DEFAULT_ENVIRONMENT) -> np.ndarray:
        f = np.asarray(frequencies, float)
        omega = 2 * np.pi * f
        s_port = math.pi * self.inner_radius_m ** 2
        m_ap = env.density_kg_m3 * self.effective_length_m / s_port
        return self.resistance_pa_s_m3 + 1j * omega * m_ap


@dataclass(frozen=True)
class DriverMount:
    driver_id: str
    centre_m: tuple[float, float]
    control: ActiveFilter = ActiveFilter()


@dataclass(frozen=True)
class Design:
    """Loudspeaker enclosure geometry and component layout."""
    id: str
    outer_dimensions_m: tuple[float, float, float]
    wall_thickness_m: float
    mounts: tuple[DriverMount, ...]
    ports: tuple[Port, ...] = ()
    leak_conductance_m3_pa_s: float = 0.0


def validate_design(design: Design, catalog: dict[str, Driver]) -> tuple[str, ...]:
    errors = []
    inner = [x - 2 * design.wall_thickness_m for x in design.outer_dimensions_m]
    if any(x <= 0 for x in inner):
        errors.append("nonpositive internal dimensions")
    gross_volume = math.prod(design.outer_dimensions_m)
    displaced = 0.0
    for mount in design.mounts:
        d = catalog.get(mount.driver_id)
        if d is None:
            errors.append(f"unknown driver: {mount.driver_id}")
        else:
            displaced += d.displaced_volume_m3
    for p in design.ports:
        displaced += math.pi * p.outer_radius_m ** 2 * p.length_m
    if gross_volume - displaced <= 0:
        errors.append("nonpositive net volume")
    return tuple(errors)


def net_volume_m3(design: Design, catalog: dict[str, Driver]) -> float:
    errs = validate_design(design, catalog)
    if errs:
        raise ValueError("; ".join(errs))
    inner = [x - 2 * design.wall_thickness_m for x in design.outer_dimensions_m]
    return (
        math.prod(inner)
        - sum(catalog[m.driver_id].displaced_volume_m3 for m in design.mounts)
        - sum(math.pi * p.outer_radius_m ** 2 * p.length_m for p in design.ports)
    )


@dataclass(frozen=True)
class SourceResult:
    status: Literal["ok", "invalid", "unsupported"]
    frequencies_hz: np.ndarray | None = None
    voltage_v_rms: np.ndarray | None = None
    current_a_rms: np.ndarray | None = None
    cone_velocity_m_s_rms: np.ndarray | None = None
    cone_volume_velocity_m3_s_rms: np.ndarray | None = None
    port_outward_volume_velocity_m3_s_rms: np.ndarray | None = None
    input_power_w: np.ndarray | None = None


def evaluate_source(
    design: Design,
    catalog: dict[str, Driver],
    frequencies: Sequence[float],
    environment: Environment = DEFAULT_ENVIRONMENT,
    reference_voltage_v_rms: float = 1.0,
) -> SourceResult:
    """Solve coupled electro-mechano-acoustic matrix equations for a vented or sealed cabinet."""
    f = np.asarray(frequencies, float)
    drivers = [catalog[m.driver_id] for m in design.mounts]
    n, nport, nf = len(drivers), len(design.ports), len(f)

    voltage = np.asarray([m.control.response(f) * reference_voltage_v_rms for m in design.mounts])
    current = np.empty((n, nf), complex)
    velocity = np.empty((n, nf), complex)
    port_in = np.empty((nport, nf), complex)
    input_power = np.empty(nf)

    cb = net_volume_m3(design, catalog) / (environment.density_kg_m3 * environment.sound_speed_m_s ** 2)

    for k, freq in enumerate(f):
        s = 2j * np.pi * freq
        size = 2 * n + 1 + nport
        a = np.zeros((size, size), complex)
        b = np.zeros(size, complex)

        for j, driver in enumerate(drivers):
            ze = driver.re_ohm + s * driver.le_h
            zm = driver.resistance_n_s_m + s * driver.moving_mass_kg + 1.0 / (s * driver.compliance_m_n)
            a[j, j], a[j, n + j], b[j] = ze, driver.bl_n_a, voltage[j, k]
            a[n + j, j], a[n + j, n + j] = -driver.bl_n_a, zm
            a[n + j, 2 * n] = driver.piston_area_m2
            a[2 * n, n + j] = driver.piston_area_m2

        a[2 * n, 2 * n] = -(s * cb + design.leak_conductance_m3_pa_s)

        for j, port in enumerate(design.ports):
            idx = 2 * n + 1 + j
            a[2 * n, idx], a[idx, 2 * n] = -1, -1
            a[idx, idx] = port.impedance([freq], environment)[0]

        row_scale = np.max(np.abs(a), axis=1)
        ar, br = a / row_scale[:, None], b / row_scale
        col_scale = np.max(np.abs(ar), axis=0)
        ae = ar / col_scale[None, :]

        x = np.linalg.solve(ae, br) / col_scale
        current[:, k], velocity[:, k] = x[:n], x[n : 2 * n]
        port_in[:, k] = x[2 * n + 1 :]
        input_power[k] = np.real(np.vdot(current[:, k], voltage[:, k]))

    cone_flow = np.asarray([d.piston_area_m2 for d in drivers])[:, None] * velocity
    return SourceResult(
        "ok",
        frequencies_hz=f,
        voltage_v_rms=voltage,
        current_a_rms=current,
        cone_velocity_m_s_rms=velocity,
        cone_volume_velocity_m3_s_rms=cone_flow,
        port_outward_volume_velocity_m3_s_rms=-port_in,
        input_power_w=input_power,
    )

