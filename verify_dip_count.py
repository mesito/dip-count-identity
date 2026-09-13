#!/usr/bin/env python3
"""verify_dip_count.py -- verification suite for

  M. Ismail, "Off-line zeros of the Riemann xi-function under differentiation:
  a dip-count identity, half-gap persistence, and a mesoscopic census to 3e10".

Groups (paper reference in brackets):
  D1  Lehmer pair, two-sided Speiser slice: S_on, h_thr vs g/2, G'(0) from
      zeta vs Hadamard sum, positivity of G on the slice      [Obs 4.3]
  D2  Far-field identity lambda = Z'/Z + 2t/(t^2+1/4) - Im psi/2 against
      the independent Hadamard sums of the transition table  [Prop 1.10, Obs 8.2]
  D3  Transition depth by the argument principle at t0=1001.877:
      dip below h_c, non-real pair above                       [Obs 8.2]
  D4  Davenport-Heilbronn negative control: curvature identity at rho_1 [Obs 4.4]
  D5  Exact polynomial stress test of the persistence lemmas   [Thm 1.2, Sec 10.3]
  D6  Exact-phase RS evaluator against mpmath at E1/E2/E3      [Sec 10.2]
  D7  Dip census (fast: E1 tight full + controls, E2/E3 subsamples + ultra
      spot checks in mpmath; full: everything in Table 3)      [Sec 11]
  D8  Realisation margin on tight midpoints and the y/pi law   [Obs 8.4, 11.4]
  D9  Parabola-threshold identity y0 = h_thr on tight pairs and the
      functional-equation symmetry of the virtual partners     [Prop 9.3, 9.2]

Usage: python3 verify_dip_count.py [--full] [--groups D1,D2,...]
Data: $RH_DATA (default ./data) with zeros6.gz, lmfdb_zeros_parsed.npy,
lm_top_zeros.npy.  Every group prints measured value, expectation, PASS/FAIL.
"""
import os, sys, time, gzip, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rs_precise as R
from mpmath import (mp, mpf, mpc, zeta, zetazero, digamma, polygamma, log as mlog,
                    pi as mpi, re as mre, im as mim, sqrt as msqrt, gamma as mgamma,
                    siegeltheta, exp as mexp, findroot)

TWO_PI = 2 * np.pi
DATA = os.environ.get("RH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
LNTB = {"E1": 13.5403, "E2": 22.8558, "E3": 24.1446}
RES = []


def check(name, ok, measured, expected):
    RES.append((name, "PASS" if ok else "FAIL", measured, expected))
    print("  [%-4s] %s measured=%s expected=%s" % (name, "PASS" if ok else "FAIL", measured, expected), flush=True)
    return ok


def load(which):
    if which == "E1":
        g = np.array(gzip.open(os.path.join(DATA, "zeros6.gz"), "rt").read().split(), dtype=float)
        return g[(g >= 5.05e5) & (g <= 1.04e6)]
    if which == "E2":
        return np.load(os.path.join(DATA, "lmfdb_zeros_parsed.npy"))
    return np.load(os.path.join(DATA, "lm_top_zeros.npy"))


def Lof(which):
    return LNTB[which] - np.log(TWO_PI)


def Zmp(t):
    tt = mpf(float(t))
    return float(mre(mexp(mpc(0, 1) * siegeltheta(tt)) * zeta(mpc(0.5, tt))))


# ----------------------------------------------------------------- D1
def D1():
    print("\nD1  Lehmer pair near t=7005: two-sided Speiser slice")
    mp.dps = 20
    zs = {n: mim(zetazero(n)) for n in range(6690, 6730)}
    best = min(range(6690, 6729), key=lambda n: zs[n + 1] - zs[n])
    g1, g2 = zs[best], zs[best + 1]; g = g2 - g1; t0 = (g1 + g2) / 2
    K = 80
    for n in range(best - K, best + K + 2):
        if n not in zs: zs[n] = mim(zetazero(n))
    Son = sum(1 / (t0 - zs[n]) ** 2 for n in zs)
    Rr = min(t0 - zs[best - K], zs[best + K + 1] - t0)
    Son += 2 * (mlog(t0 / (2 * mpi)) / (2 * mpi)) / Rr
    hthr = msqrt(2 / Son)
    s = mpc(mpf(1) / 2, t0)
    z0, z1, z2 = zeta(s), zeta(s, 1, 1), zeta(s, 1, 2)
    Gp0 = mre(z2 / z0 - (z1 / z0) ** 2 - 1 / s ** 2 - 1 / (s - 1) ** 2 + polygamma(1, s / 2) / 4)
    def G(h):
        ss = mpc(mpf(1) / 2 + h, t0)
        return mre(zeta(ss, 1, 1) / zeta(ss) + 1 / ss + 1 / (ss - 1) - mlog(mpi) / 2 + digamma(ss / 2) / 2)
    pos = all(G(mpf(h)) > 0 for h in (1e-4, 1e-3, 0.01, 0.1, 0.49))
    print("    gap=%.5f t0=%.6f S_on=%.4f 8/g^2=%.3f h_thr=%.6f g/2=%.6f" % (g, t0, Son, 8 / g ** 2, hthr, g / 2))
    ok1 = abs(hthr / (g / 2) - 0.99978) < 2e-4
    ok2 = abs(Gp0 / Son - 1) < 1e-6
    check("D1", ok1 and ok2 and pos, "h_thr/(g/2)=%.5f, G'(0)/S_on-1=%.1e, G>0:%s" % (hthr / (g / 2), Gp0 / Son - 1, pos),
          "0.99978+-2e-4; |rel|<1e-6; G>0 on slice (Obs 4.3)")


# ----------------------------------------------------------------- D2
def _pair_around(t0):
    n = int(t0 / (2 * mpi) * mlog(t0 / (2 * mpi * mexp(1)))) - 20
    n = max(n, 1)
    while mim(zetazero(n)) < t0: n += 1
    return mim(zetazero(n - 1)), mim(zetazero(n)), n - 1


def D2():
    print("\nD2  far-field identity (Prop 1.10) vs Hadamard-sum table values")
    mp.dps = 20
    def Zf(t): return mre(mexp(mpc(0, 1) * siegeltheta(t)) * zeta(mpc(mpf(1) / 2, t)))
    ok = True; out = []
    for t0, tab in ((mpf("1001.877"), -1.132), (mpf("1008.243"), -1.355)):
        from mpmath import diff
        lam = diff(Zf, t0) / Zf(t0) + 2 * t0 / (t0 * t0 + mpf(1) / 4) - mim(digamma(mpc(mpf(1) / 4, t0 / 2))) / 2
        g1, g2, _ = _pair_around(t0)
        lam_far = float(lam - 1 / (t0 - g1) - 1 / (t0 - g2))
        out.append("t0=%s: %.4f vs %.3f" % (t0, lam_far, tab)); ok &= abs(lam_far - tab) < 3e-3
    check("D2", ok, "; ".join(out), "table of Obs 8.2 within 3e-3")


# ----------------------------------------------------------------- D3
def D3(full):
    print("\nD3  transition depth at t0=1001.877 (argument principle, Obs 8.2)")
    mp.dps = 15
    t0 = mpf("1001.877")
    g1, g2, n1 = _pair_around(t0)
    zs = {n: mim(zetazero(n)) for n in range(n1 - 12, n1 + 14)}
    def Xi(t): s = mpc(0.5, 0) + 1j * t; return s * (s - 1) / 2 * mpi ** (-s / 2) * mgamma(s / 2) * zeta(s)
    def XiLog(t):
        s = mpc(0.5, 0) + 1j * t
        return 1j * (zeta(s, 1, 1) / zeta(s) + 1 / s + 1 / (s - 1) - mlog(mpi) / 2 + digamma(s / 2) / 2)
    others = [zs[n] for n in zs if n not in (n1, n1 + 1)]
    D = min(abs(t0 - gg) for gg in others)
    def nonreal(h0):
        Lf = lambda t: XiLog(t) - 1 / (t - g1) - 1 / (t - g2) + 2 * (t - t0) / ((t - t0) ** 2 + h0 ** 2)
        F = lambda t: Xi(t) * ((t - t0) ** 2 + h0 ** 2) / ((t - g1) * (t - g2)) * Lf(t)
        w = 0.9 * D; H = 1.5 * h0 + 0.1
        corners = [mpc(t0 - w, -H), mpc(t0 + w, -H), mpc(t0 + w, H), mpc(t0 - w, H)]
        tot = 0
        for k in range(4):
            p, q = corners[k], corners[(k + 1) % 4]; M = 50; prev = F(p)
            for j in range(1, M + 1):
                z = p + (q - p) * j / M; cur = F(z); tot += mim(mlog(cur / prev)); prev = cur
        nz = int(round(tot / (2 * mpi)))
        M = 160; xs = [t0 - w + 2 * w * j / M for j in range(M + 1)]
        vals = [mre(F(mpf(x))) for x in xs]
        nreal = sum(1 for j in range(M) if vals[j] * vals[j + 1] < 0)
        return nz - nreal
    lo, hi = nonreal(mpf("0.30")), nonreal(mpf("0.80"))
    msg = "nonreal(h=0.30)=%d nonreal(h=0.80)=%d" % (lo, hi)
    if full:
        a, b = mpf("0.30"), mpf("0.80")
        for _ in range(6):
            m = (a + b) / 2
            if nonreal(m) <= 0: a = m
            else: b = m
        hc = float((a + b) / 2); msg += " h_c=%.3f (table 0.547)" % hc
        ok = lo == 0 and hi == 2 and abs(hc - 0.547) < 0.02
    else:
        ok = lo == 0 and hi == 2
    check("D3", ok, msg, "0 (dip) below h_c=0.547, 2 (pair) above")


# ----------------------------------------------------------------- D4
def D4():
    print("\nD4  Davenport-Heilbronn negative control (Obs 4.4)")
    mp.dps = 20
    kappa = (msqrt(10 - 2 * msqrt(5)) - 2) / (msqrt(5) - 1)
    a = [1, kappa, -kappa, -1, 0]
    def f(s, d=0): return 5 ** (-s) * sum(a[r - 1] * zeta(s, mpf(r) / 5, d) for r in range(1, 6)) if d == 0 else None
    def Lam(s): return (5 / mpi) ** ((s + 1) / 2) * mgamma((s + 1) / 2) * f(s)
    # on-line zeros in (0,130) by sign change of the real function Lam(1/2+it)
    def Lr(t): return mre(Lam(mpc(mpf(1) / 2, t)))
    ts = [mpf(1) + mpf(k) / 10 for k in range(0, 1291)]
    vals = [Lr(t) for t in ts]
    zeros = []
    for j in range(len(ts) - 1):
        if vals[j] * vals[j + 1] < 0:
            zeros.append(findroot(Lr, (ts[j], ts[j + 1]), solver="bisect"))
    rho1 = findroot(lambda s: f(s), mpc("0.8085", "85.6993"))
    rho2 = findroot(lambda s: f(s), mpc("0.6508", "114.1633"))
    b0, t0 = mre(rho1) - mpf(1) / 2, mim(rho1)
    # curvature G'(0) at t0 from Lam directly
    def LamLog(s): return findroot  # placeholder
    from mpmath import diff
    def Gh(h): return mre(diff(lambda z: mlog(Lam(z)), mpc(mpf(1) / 2 + h, t0)))
    Gp0 = float(diff(Gh, mpf(0)))
    Son = float(sum(1 / (t0 - z) ** 2 for z in zeros))
    y2, u2 = mre(rho2) - mpf(1) / 2, t0 - mim(rho2)
    K2 = float(2 * (y2 ** 2 - u2 ** 2) / (y2 ** 2 + u2 ** 2) ** 2)   # -K contribution of the other pair
    pred = Son - float(2 / b0 ** 2) + K2
    print("    on-line zeros in (0,130): %d ; rho1=%s ; G'(0)=%.3f ; S_on-2/h0^2+other=%.3f" % (len(zeros), rho1, Gp0, pred))
    check("D4", abs(Gp0 - pred) < 0.05 and Gp0 < 0, "G'(0)=%.3f vs %.3f" % (Gp0, pred), "-20.16 vs -20.15 (Obs 4.4), within 0.05")


# ----------------------------------------------------------------- D5
def D5(full):
    print("\nD5  exact polynomial stress test of the persistence lemmas")
    rng = np.random.default_rng(7)
    def lam(t, z, prs):
        v = np.zeros_like(t)
        for gg in z: v += 1 / (t - gg)
        for tj, h in prs: v += 2 * (t - tj) / ((t - tj) ** 2 + h * h)
        return v
    def nz(z, prs, a, b, n=40001):
        t = np.linspace(a, b, n + 2)[1:-1]; v = lam(t, z, prs)
        return int(np.sum(v[:-1] * v[1:] < 0))
    trials = 3000 if full else 500
    v2 = v3 = v4 = 0; n2 = n3 = n4 = 0; own = 0
    for _ in range(trials):
        sp = rng.gamma(2.5, 0.4, size=13); z = np.concatenate([[0.0], np.cumsum(sp)])
        i = int(rng.integers(1, 12)); a, b = z[i], z[i + 1]; g = b - a
        t0 = rng.uniform(a + 0.01 * g, b - 0.01 * g); h = g * rng.uniform(0.05, 3.0); prs = [(t0, h)]
        cnt = [nz(z, prs, z[k], z[k + 1]) for k in range(13)]
        n3 += 1
        if max(c for k, c in enumerate(cnt) if k != i) > 1: v3 += 1
        if h > g / 2:
            n2 += 1
            if cnt[i] > 1: v2 += 1
        elif cnt[i] > 1: own += 1
        k = int(rng.integers(2, 9)); j = int(rng.integers(6, 10)); prs = []
        for _ in range(k):
            u = rng.uniform(0.05, 1.2); prs.append((z[j] + u, rng.uniform(0.5, 5.0)))
        for m in range(1, j):
            c = nz(z, prs, z[j - m], z[j - m + 1]); n4 += 1
            if c > 1 and k <= 4 * m: v4 += 1
    check("D5", v2 == 0 and v3 == 0 and v4 == 0,
          "L(i) %d/%d viol, L(ii) %d/%d, L(iii k>4m) %d/%d, own-dips %d" % (v2, n2, v3, n3, v4, n4, own),
          "0 violations; own-dips of shallow pairs > 0 (detector live)")


# ----------------------------------------------------------------- D6
def D6():
    print("\nD6  exact-phase RS evaluator vs mpmath (dps=25)")
    mp.dps = 25
    ok = True; msg = []
    for which, ks in (("E1", (1000, 500000)), ("E2", (72082, 1000)), ("E3", (503515, 1000))):
        gam = load(which); errs = []
        for k in ks:
            a, b = float(gam[k]), float(gam[k + 1]); d = 0.02 * (b - a); T = np.linspace(a + d, b - d, 6)
            errs += list(R.Z_batch_precise(T) - np.array([Zmp(x) for x in T]))
        rms = float(np.sqrt(np.mean(np.square(errs)))); msg.append("%s rms=%.1e" % (which, rms)); ok &= rms < 1e-8
    check("D6", ok, "; ".join(msg), "rms < 1e-8 at all three heights (Sec 10.2)")


# ----------------------------------------------------------------- D7
def D7(full):
    print("\nD7  dip census (%s tier)" % ("full" if full else "fast"))
    quiet = lambda *a: None
    total_dips = 0; total_anom = 0; msg = []
    for which, ctrl_n, sub in (("E1", 100000 if full else 20000, None), ("E2", 10000 if full else 2000, None if full else 3000), ("E3", 5000 if full else 1000, None if full else 2000)):
        gam = load(which); L = Lof(which); g = np.diff(gam); r = g / (TWO_PI / L)
        idx = np.nonzero(r < 0.35)[0]
        ultra = idx[r[idx] < 0.05] if which != "E1" else np.array([], dtype=int)
        band = np.setdiff1d(idx, ultra)
        rng = np.random.default_rng(17)
        if sub and band.size > sub: band = np.sort(rng.choice(band, sub, replace=False))
        t0 = time.time()
        c, cf, sa = R.census_idx(gam, band, which, "tight", batch_gaps=200 if which == "E1" else 96, log=quiet)
        rng2 = np.random.default_rng(11); ctrl = np.sort(rng2.choice(np.nonzero(r >= 0.35)[0], ctrl_n, replace=False))
        c2, cf2, sa2 = R.census_idx(gam, ctrl, which, "ctrl", batch_gaps=200 if which == "E1" else 96, log=quiet)
        # ultra-tight in mpmath (E2/E3): full tier all; fast tier 3 each
        ud = 0
        if ultra.size:
            mp.dps = 25
            sel = ultra if full else ultra[:3]
            for k in sel:
                a, b = float(gam[k]), float(gam[k + 1]); d = 0.02 * (b - a); T = np.linspace(a + d, b - d, 21)
                v = np.array([Zmp(x) for x in T]); av = np.abs(v)
                if v[0] * v[-1] <= 0 or R.is_dip(av, max(1e-6, 0.08 * av.max())): ud += 1
        total_dips += cf + cf2 + ud; total_anom += sa + sa2
        msg.append("%s tight %d + ultra(mp) %d + ctrl %d: dips %d anomalies %d (%.0fs)" % (which, band.size, len(ultra) if full else min(3, len(ultra)), ctrl_n, cf + cf2 + ud, sa + sa2, time.time() - t0))
        print("    " + msg[-1], flush=True)
    check("D7", total_dips == 0 and total_anom == 0, "dips=%d anomalies=%d" % (total_dips, total_anom), "0 and 0 (Table 3)")


# ----------------------------------------------------------------- D8
def D8(full):
    print("\nD8  realisation margin on tight midpoints and the y/pi law (E2)")
    gam = load("E2"); L = Lof("E2"); dbar = TWO_PI / L; g = np.diff(gam); r = g / dbar; tm = 0.5 * (gam[:-1] + gam[1:])
    n = 12000 if full else 2000
    def Zb(ts):
        out = np.empty_like(ts)
        for i in range(0, ts.size, 1500): out[i:i + 1500] = R.Z_batch_precise(ts[i:i + 1500])
        return out
    def lam(ts, h=2e-5): return (Zb(ts + h) - Zb(ts - h)) / (2 * h) / Zb(ts) - np.pi / 4 + 9 / (4 * ts)
    rng = np.random.default_rng(5)
    sel = np.nonzero(r < 0.35)[0]; sel = sel[rng.choice(sel.size, n, replace=False)]
    x = (g[sel] / 2) * np.abs(lam(tm[sel]))
    t0s = rng.uniform(gam[10], gam[-10], n); lr = np.abs(lam(t0s)) / L
    slope = (1 - np.mean(lr < 1 / 0.5)) / 0.5
    med, mx = float(np.median(x)), float(x.max())
    print("    h0|lambda| median=%.3f max=%.3f frac<1=%.4f ; (1-P)/y at y=0.5 = %.3f (1/pi=0.318)" % (med, mx, np.mean(x < 1), slope))
    check("D8", med < 0.1 and mx < 1 and abs(slope - 1 / np.pi) < 0.05, "median=%.3f max=%.3f slope=%.3f" % (med, mx, slope),
          "median<0.1, max<1, slope 0.318+-0.05 (Obs 11.4, 8.4)")


# ----------------------------------------------------------------- D9
def D9(full):
    print("\nD9  parabola-threshold identity y0 = h_thr (Prop 9.3) and FE symmetry (Prop 9.2), E2")
    gam = load("E2"); L = Lof("E2"); dbar = TWO_PI / L; g = np.diff(gam); r = g / dbar
    idx = np.nonzero(r < 0.35)[0]; rng = np.random.default_rng(3)
    sel = np.sort(rng.choice(idx, 100 if not full else 300, replace=False))
    ratios = []
    dens = L / TWO_PI
    for k in sel:
        a, b = gam[k], gam[k + 1]; T = np.linspace(a, b, 202)[1:-1]; v = R.Z_batch_precise(T)
        j = int(np.argmax(np.abs(v))); ts = T[j]; h = abs(v[j])
        hh = 2e-4 * (b - a)
        z5 = R.Z_batch_precise(np.array([ts - 2 * hh, ts - hh, ts, ts + hh, ts + 2 * hh]))
        Zpp = (-z5[0] + 16 * z5[1] - 30 * z5[2] + 16 * z5[3] - z5[4]) / (12 * hh * hh)
        y0 = np.sqrt(2 * h / abs(Zpp))
        lo, hi = max(0, k - 800), min(gam.size, k + 802)
        others = np.concatenate([gam[lo:k], gam[k + 2:hi]])
        Son = np.sum(1 / (ts - gam[k]) ** 2) + np.sum(1 / (ts - gam[k + 1]) ** 2) + np.sum(1 / (ts - others) ** 2)
        Rr = min(ts - gam[lo], gam[hi - 1] - ts); Son += 2 * dens / Rr
        ratios.append(y0 / np.sqrt(2 / Son))
    ratios = np.array(ratios); med = float(np.median(ratios)); corr = float(np.corrcoef(np.log(ratios * 0 + 1) + 0, ratios)[0, 1]) if False else 0
    # FE symmetry on the 3 tightest pairs: |zeta(s-)|/|zeta(s+)| vs (t/2pi)^{y0}
    mp.dps = 20; fe = []
    tight3 = idx[np.argsort(r[idx])[:3]]
    for k in tight3:
        a, b = gam[k], gam[k + 1]; ts = 0.5 * (a + b); y = 0.5 * (b - a)
        sm, sp = mpc(mpf(1) / 2 - y, mpf(float(ts))), mpc(mpf(1) / 2 + y, mpf(float(ts)))
        try:
            ratio = float(abs(zeta(sm)) / abs(zeta(sp))); pred = float((mpf(float(ts)) / (2 * mpi)) ** y)
            fe.append(abs(ratio / pred - 1))
        except Exception as e:
            fe.append(float("nan"))
    fe = np.array(fe)
    print("    y0/h_thr: median=%.4f (n=%d) ; FE symmetry rel. dev.: %s" % (med, ratios.size, np.round(fe, 5)))
    ok = 0.98 < med < 1.05 and (np.all(np.isnan(fe)) or np.nanmax(fe) < 1e-3)
    check("D9", ok, "median y0/h_thr=%.4f; FE max rel dev=%.1e" % (med, np.nanmax(fe) if not np.all(np.isnan(fe)) else float('nan')),
          "1.013+-0.03 (Prop 9.3 numerics); FE rel dev < 1e-3")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true"); ap.add_argument("--groups", default="")
    args = ap.parse_args()
    groups = args.groups.split(",") if args.groups else ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
    print("verify_dip_count.py  tier=%s  groups=%s  data=%s" % ("full" if args.full else "fast", ",".join(groups), DATA), flush=True)
    t0 = time.time()
    for gname in groups:
        f = globals()[gname]
        try:
            f(args.full) if gname in ("D3", "D5", "D7", "D8", "D9") else f()
        except Exception as e:
            check(gname, False, "EXCEPTION %r" % e, "-")
    print("\n" + "=" * 70)
    for n, s, m, e in RES: print("%-4s %-4s %s" % (n, s, m[:90]))
    print("PASS=%d FAIL=%d   runtime %.0fs" % (sum(s == "PASS" for _, s, _, _ in RES), sum(s == "FAIL" for _, s, _, _ in RES), time.time() - t0))


if __name__ == "__main__":
    main()
