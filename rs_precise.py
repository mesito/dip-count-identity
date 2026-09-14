#!/usr/bin/env python3
"""rs_precise.py -- exact-phase Riemann-Siegel evaluator for Z(t) at t up to ~1e12,
with the dip detector used by the census (paper Sections 10-11).

Z_batch_precise(ts): RS main sum + Gabcke C0 and C1 terms with exact phase
reduction: theta(T0) mod 2pi from mpmath once per 50-wide t-window + Taylor in
delta = t - T0; ln n as double-double; T0*ln n reduced with exact fmod and a
two-double 2pi. Verified against mpmath (suite group D6, 40 points per ensemble): rms
2.2e-9 (t~7e5), 6.7e-11 (8.4e9), 4.6e-10 (3.1e10).
"""
import time
import numpy as np

TWO_PI = 2.0 * np.pi
TWO_PI_LO = 2.4492935982947064e-16   # 2pi - float64(2pi)

def theta(t):
    return 0.5 * t * np.log(t / TWO_PI) - 0.5 * t - np.pi / 8.0 \
        + 1.0 / (48.0 * t) + 7.0 / (5760.0 * t ** 3)


def is_dip(v, prom):
    """True iff v (|Z| on the interior grid) has two local maxima separated
    by a saddle at least prom below the LOWER of the two peaks (true
    saddle prominence, not flank-to-edge)."""
    peaks = [i for i in range(1, len(v) - 1)
             if v[i] > v[i - 1] and v[i] > v[i + 1]]
    for x in range(len(peaks)):
        for y in range(x + 1, len(peaks)):
            i, j = peaks[x], peaks[y]
            saddle = v[i + 1:j].min() if j > i + 1 else min(v[i], v[j])
            if min(v[i], v[j]) - saddle >= prom:
                return True
    return False


def census_idx(gam, idx, name, label, mA=12, mB=64, batch_gaps=96, log=print):
    g = np.diff(gam)
    if True:
        t0 = time.time()
        cand = []
        shallow = []
        signbad = 0; signk = []
        for b0 in range(0, idx.size, batch_gaps):
            ii = idx[b0:b0 + batch_gaps]
            # build grid: margins 2% g, mA interior points + 2 margin pts
            grids = []
            for k in ii:
                a, b = gam[k], gam[k + 1]
                d = 0.02 * (b - a)
                grids.append(np.linspace(a + d, b - d, mA + 2))
            T = np.concatenate(grids)
            Zv = Z_batch_precise(T)
            for j, k in enumerate(ii):
                v = Zv[j * (mA + 2):(j + 1) * (mA + 2)]
                if v[0] * v[-1] <= 0:
                    signbad += 1; signk.append(int(k))
                    continue
                av = np.abs(v)
                if av.max() < 1e-6:
                    shallow.append(k); continue
                prom = max(1e-6, 1e-3 * av.max())
                if is_dip(av, prom):
                    cand.append(k)
            if (b0 // batch_gaps) % 40 == 39:
                done = min(b0 + batch_gaps, idx.size)
                log("    %s %s: %d/%d gaps (%.0fs), candidates so far: %d"
                    % (name, label, done, idx.size, time.time() - t0,
                       len(cand)))
        # Stage B refine
        conf = []
        for k in cand:
            a, b = gam[k], gam[k + 1]
            d = 0.02 * (b - a)
            T = np.linspace(a + d, b - d, mB + 2)
            v = Z_batch_precise(T)
            av = np.abs(v)
            prom = max(1e-6, 1e-3 * av.max())
            if v[0] * v[-1] > 0 and is_dip(av, prom):
                conf.append(k)
        log("  %s %s: gaps=%d  stageA candidates=%d  CONFIRMED DIPS=%d  "
            "sign-anomalies=%d  RS-blind-shallow=%d  (%.0fs)"
            % (name, label, idx.size, len(cand), len(conf), signbad,
               len(shallow), time.time() - t0))
        if shallow:
            log("      shallow k-list: %s" % [int(x) for x in shallow])
        if signk:
            log("      sign-anomaly k-list: %s" % signk)
        for k in conf[:20]:
            log("      dip at gap k=%d, gamma=[%.6f, %.6f], g=%.5f"
                % (k, gam[k], gam[k + 1], g[k]))
    return len(cand), len(conf), signbad


_LNCACHE = {}
def _ln_dd(Nmax):
    """ln n for 1..Nmax as (hi, lo) double-double from mpmath (cached)."""
    if Nmax not in _LNCACHE:
        from mpmath import mp, log as mlog, mpf
        mp.dps = 34
        hi = np.log(np.arange(1, Nmax + 1, dtype=np.float64))
        lo = np.array([float(mlog(mpf(n)) - mpf(float(h)))
                       for n, h in zip(range(1, Nmax + 1), hi)])
        _LNCACHE[Nmax] = (hi, lo)
    return _LNCACHE[Nmax]

def Z_batch_precise(ts, n_chunk=8192, window=50.0):
    """Exact-phase RS; the theta Taylor expansion about T0 needs |t-T0| <= ~25,
    so inputs spanning more than `window` are split into t-windows."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.size and (ts.max() - ts.min()) > window:
        out = np.empty_like(ts)
        keys = np.floor(ts / window)
        for kv in np.unique(keys):
            m = keys == kv
            out[m] = Z_batch_precise(ts[m], n_chunk, window)
        return out
    return _Z_batch_precise_core(ts, n_chunk)


def _Z_batch_precise_core(ts, n_chunk=8192):
    from mpmath import mp, mpf, siegeltheta, pi as mpi
    ts = np.asarray(ts, dtype=np.float64)
    T0 = float(np.floor(ts.mean()))
    delta = ts - T0
    a = np.sqrt(ts / TWO_PI)
    N = np.floor(a).astype(np.int64)
    Nmin, Nmax = int(N.min()), int(N.max())
    mp.dps = 30
    th0 = float(siegeltheta(mpf(T0)) % (2 * mpi))
    thp = 0.5 * np.log(T0 / TWO_PI)
    thpp = 0.5 / T0
    th = th0 + thp * delta + 0.5 * thpp * delta * delta
    s = np.zeros_like(ts)
    LH, LL = _ln_dd(Nmax)
    n0 = 1
    while n0 <= Nmin:
        n1 = min(n0 + n_chunk - 1, Nmin)
        n = np.arange(n0, n1 + 1, dtype=np.float64)
        ln = LH[n0 - 1:n1]
        lhi = np.round(ln * 8192.0) / 8192.0
        llo = (ln - lhi) + LL[n0 - 1:n1]
        x = T0 * lhi                                    # exact
        rem = np.remainder(x, TWO_PI)                    # exact fmod
        q = np.round((x - rem) / TWO_PI)                 # exact integer
        base = rem - q * TWO_PI_LO + T0 * llo            # true-2pi reduction
        base = np.remainder(base, TWO_PI)
        w = 1.0 / np.sqrt(n)
        ph = th[:, None] - base[None, :] - delta[:, None] * ln[None, :]
        s += np.cos(ph) @ w
        n0 = n1 + 1
    if Nmax > Nmin:
        for j in np.nonzero(N > Nmin)[0]:
            n = np.arange(Nmin + 1, N[j] + 1, dtype=np.float64)
            ln = LH[Nmin:N[j]]; lhi = np.round(ln * 8192.0) / 8192.0
            llo = (ln - lhi) + LL[Nmin:N[j]]
            x = T0 * lhi; rem = np.remainder(x, TWO_PI); q = np.round((x - rem) / TWO_PI)
            base = np.remainder(rem - q * TWO_PI_LO + T0 * llo, TWO_PI)
            s[j] += (np.cos(th[j] - base - delta[j] * ln) / np.sqrt(n)).sum()
    p = a - N
    def Psi(x):
        return np.cos(TWO_PI * (x * x - x - 1.0 / 16.0)) / np.cos(TWO_PI * x)
    hh = 2e-3
    Psi3 = (Psi(p + 2*hh) - 2*Psi(p + hh) + 2*Psi(p - hh) - Psi(p - 2*hh)) / (2 * hh**3)
    C1 = -Psi3 / (96.0 * np.pi ** 2)
    tau = ts / TWO_PI
    corr = np.where(N % 2 == 1, 1.0, -1.0) * tau ** -0.25 * (Psi(p) + C1 * tau ** -0.5)
    return 2.0 * s + corr


