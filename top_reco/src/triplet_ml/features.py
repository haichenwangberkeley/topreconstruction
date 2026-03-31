#!/usr/bin/env python3
"""Physics feature construction and sanity checks for GenJet triplets."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np

FEATURE_COLUMNS = (
    "dr_ab",
    "dr_ac",
    "dr_bc",
    "mij_over_m123_ab",
    "mij_over_m123_ac",
    "mij_over_m123_bc",
)

OBSERVABLE_COLUMNS = (
    "m123",
    "mij_ab",
    "mij_ac",
    "mij_bc",
    "triplet_pt",
    "triplet_eta",
    "triplet_phi",
)


def _wrap_phi(delta_phi: float) -> float:
    return math.atan2(math.sin(delta_phi), math.cos(delta_phi))


def _delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    deta = eta1 - eta2
    dphi = _wrap_phi(phi1 - phi2)
    return math.hypot(deta, dphi)


def _massless_fourvec(pt: float, eta: float, phi: float) -> Tuple[float, float, float, float]:
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    e = math.sqrt(max(px * px + py * py + pz * pz, 0.0))
    return px, py, pz, e


def _inv_mass(px: float, py: float, pz: float, e: float) -> float:
    m2 = e * e - (px * px + py * py + pz * pz)
    if m2 < 0.0:
        m2 = 0.0
    return math.sqrt(m2)


def _triplet_kinematics(px: float, py: float, pz: float, e: float) -> Tuple[float, float, float, float]:
    pt = math.hypot(px, py)
    eta = math.asinh(pz / pt) if pt > 0.0 else 0.0
    phi = math.atan2(py, px)
    mass = _inv_mass(px, py, pz, e)
    return pt, eta, phi, mass


def compute_triplet_feature_payload(
    pt: Sequence[float], eta: Sequence[float], phi: Sequence[float], i: int, j: int, k: int
) -> Dict[str, float]:
    pa = _massless_fourvec(float(pt[i]), float(eta[i]), float(phi[i]))
    pb = _massless_fourvec(float(pt[j]), float(eta[j]), float(phi[j]))
    pc = _massless_fourvec(float(pt[k]), float(eta[k]), float(phi[k]))

    dr_ab = _delta_r(float(eta[i]), float(phi[i]), float(eta[j]), float(phi[j]))
    dr_ac = _delta_r(float(eta[i]), float(phi[i]), float(eta[k]), float(phi[k]))
    dr_bc = _delta_r(float(eta[j]), float(phi[j]), float(eta[k]), float(phi[k]))

    px_ab = pa[0] + pb[0]
    py_ab = pa[1] + pb[1]
    pz_ab = pa[2] + pb[2]
    e_ab = pa[3] + pb[3]
    m_ab = _inv_mass(px_ab, py_ab, pz_ab, e_ab)

    px_ac = pa[0] + pc[0]
    py_ac = pa[1] + pc[1]
    pz_ac = pa[2] + pc[2]
    e_ac = pa[3] + pc[3]
    m_ac = _inv_mass(px_ac, py_ac, pz_ac, e_ac)

    px_bc = pb[0] + pc[0]
    py_bc = pb[1] + pc[1]
    pz_bc = pb[2] + pc[2]
    e_bc = pb[3] + pc[3]
    m_bc = _inv_mass(px_bc, py_bc, pz_bc, e_bc)

    px_abc = pa[0] + pb[0] + pc[0]
    py_abc = pa[1] + pb[1] + pc[1]
    pz_abc = pa[2] + pb[2] + pc[2]
    e_abc = pa[3] + pb[3] + pc[3]
    triplet_pt, triplet_eta, triplet_phi = _triplet_kinematics(px_abc, py_abc, pz_abc, e_abc)[:3]
    m_abc = _inv_mass(px_abc, py_abc, pz_abc, e_abc)

    if m_abc > 0.0:
        r_ab = m_ab / m_abc
        r_ac = m_ac / m_abc
        r_bc = m_bc / m_abc
    else:
        r_ab = 0.0
        r_ac = 0.0
        r_bc = 0.0

    return {
        "dr_ab": dr_ab,
        "dr_ac": dr_ac,
        "dr_bc": dr_bc,
        "mij_over_m123_ab": r_ab,
        "mij_over_m123_ac": r_ac,
        "mij_over_m123_bc": r_bc,
        "m123": m_abc,
        "mij_ab": m_ab,
        "mij_ac": m_ac,
        "mij_bc": m_bc,
        "triplet_pt": triplet_pt,
        "triplet_eta": triplet_eta,
        "triplet_phi": triplet_phi,
    }


def compute_triplet_features(
    pt: Sequence[float], eta: Sequence[float], phi: Sequence[float], i: int, j: int, k: int
) -> Tuple[float, float, float, float, float, float]:
    payload = compute_triplet_feature_payload(pt, eta, phi, i, j, k)
    return tuple(payload[name] for name in FEATURE_COLUMNS)  # type: ignore[return-value]


def assert_feature_values_sane(values: Sequence[float]) -> None:
    if len(values) != 6:
        raise ValueError("Feature vector must have length 6")
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
        raise ValueError(f"Non-finite feature values encountered: {values}")

    dr_ab, dr_ac, dr_bc, r_ab, r_ac, r_bc = values
    for dr in (dr_ab, dr_ac, dr_bc):
        if dr < 0.0 or dr > 20.0:
            raise ValueError(f"DeltaR outside sanity range [0,20]: {dr}")
    for ratio in (r_ab, r_ac, r_bc):
        if ratio < 0.0 or ratio > 10.0:
            raise ValueError(f"Mass ratio outside sanity range [0,10]: {ratio}")


def assert_feature_batch_sane(batch: Dict[str, np.ndarray]) -> None:
    for column in FEATURE_COLUMNS:
        if column not in batch:
            raise ValueError(f"Missing feature column in batch sanity check: {column}")
        values = np.asarray(batch[column], dtype=np.float64)
        if values.size == 0:
            continue
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values found in column {column}")
    for dr_column in ("dr_ab", "dr_ac", "dr_bc"):
        values = np.asarray(batch[dr_column], dtype=np.float64)
        if np.any(values < 0.0) or np.any(values > 20.0):
            raise ValueError(f"DeltaR sanity failure in column {dr_column}")
    for ratio_column in ("mij_over_m123_ab", "mij_over_m123_ac", "mij_over_m123_bc"):
        values = np.asarray(batch[ratio_column], dtype=np.float64)
        if np.any(values < 0.0) or np.any(values > 10.0):
            raise ValueError(f"Mass-ratio sanity failure in column {ratio_column}")


def assert_observable_batch_sane(batch: Dict[str, np.ndarray]) -> None:
    for column in OBSERVABLE_COLUMNS:
        if column not in batch:
            raise ValueError(f"Missing observable column in sanity check: {column}")
        values = np.asarray(batch[column], dtype=np.float64)
        if values.size == 0:
            continue
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite values found in observable column {column}")

    for mass_column in ("m123", "mij_ab", "mij_ac", "mij_bc"):
        values = np.asarray(batch[mass_column], dtype=np.float64)
        if np.any(values < 0.0):
            raise ValueError(f"Negative mass values found in {mass_column}")

    phi = np.asarray(batch["triplet_phi"], dtype=np.float64)
    if np.any(phi < -math.pi - 1e-6) or np.any(phi > math.pi + 1e-6):
        raise ValueError("Triplet phi outside [-pi, pi] range")
