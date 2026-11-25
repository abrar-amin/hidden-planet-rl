"""Laplace coefficients and alpha-dependent first-order resonance coefficients f_j(alpha), g_j(alpha).

LXW12 tabulate f, g at exact commensurability; away from it (|Delta| up to 0.05) they change by
tens of percent for j >= 3, which is the dominant amplitude error of the tabulated formula.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_PSI = (np.arange(2048) + 0.5) * 2 * np.pi / 2048


def laplace_b(j, alpha, s=0.5):
    return 2 * np.mean(np.cos(j * _PSI) / (1 - 2 * alpha * np.cos(_PSI) + alpha * alpha) ** s)


def laplace_db(j, alpha, h=1e-5):
    return (laplace_b(j, alpha + h) - laplace_b(j, alpha - h)) / (2 * h)


@lru_cache(maxsize=200_000)
def _fg(j, alpha_r):
    a = alpha_r
    f = -0.5 * (2 * j * laplace_b(j, a) + a * laplace_db(j, a))
    g = 0.5 * ((2 * j - 1) * laplace_b(j - 1, a) + a * laplace_db(j - 1, a))
    if j == 2:
        g -= 2 * a          # indirect term
    return f, g


def fg(j, period_ratio):
    """(f, g) for the j:j-1 term at period ratio P2/P1 (alpha = ratio^(-2/3))."""
    return _fg(int(j), round(float(period_ratio) ** (-2.0 / 3.0), 5))


def multi_resonance_ttv(n, P1, P2, mu2, z1, z2, lam2_0, js=(2, 3, 4, 5, 6), alpha_dep=True):
    """Sum of first-order near-resonant terms (each LXW-like with its own Delta_j)."""
    from .physics import F_COEF, G_COEF, TWO_PI
    n = np.asarray(n, float)
    out = np.zeros_like(n)
    for j in js:
        d = (P2 / P1) * (j - 1) / j - 1
        if abs(d) < 1e-4:
            continue
        f, g = fg(j, P2 / P1) if alpha_dep else (F_COEF[j], G_COEF[j])
        V = P1 * mu2 / (np.pi * j ** (2 / 3) * (j - 1) ** (1 / 3) * d) * (-f - 1.5 * np.conj(f * z1 + g * z2) / d)
        lamj = j * lam2_0 + TWO_PI * (-(j - 1) * d / (1 + d)) * n
        out += np.imag(V * np.exp(1j * lamj))
    return out


_TABLE: dict[int, tuple] = {}


def fg_array(j, period_ratio):
    """Vectorised (f, g) by interpolation in alpha (table built once per j)."""
    j = int(j)
    if j not in _TABLE:
        a0 = ((j - 1) / j) ** (2.0 / 3.0)
        al = np.linspace(a0 * 0.90, min(a0 * 1.10, 0.97), 81)
        fs, gs = zip(*[_fg(j, round(float(a), 5)) for a in al])
        _TABLE[j] = (al, np.array(fs), np.array(gs))
    al, fs, gs = _TABLE[j]
    a = np.asarray(period_ratio, float) ** (-2.0 / 3.0)
    return np.interp(a, al, fs), np.interp(a, al, gs)
