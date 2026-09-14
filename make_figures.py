#!/usr/bin/env python3
"""make_figures.py -- regenerates every figure of the paper from the data
(English labels). Usage: python3 make_figures.py [fig1 fig2 ...]  (default: all)
Output: ../paper/figures/*.png.  Data via $RH_DATA (see README)."""
import os, sys, gzip, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rs_precise as R

TWO_PI = 2 * np.pi
DATA = os.environ.get("RH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
LNTB = {"E1": 13.5403, "E2": 22.8558, "E3": 24.1446}
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def load(w):
    if w == "E1":
        g = np.array(gzip.open(os.path.join(DATA, "zeros6.gz"), "rt").read().split(), dtype=float)
        return g[(g >= 5.05e5) & (g <= 1.04e6)]
    return np.load(os.path.join(DATA, "lmfdb_zeros_parsed.npy" if w == "E2" else "lm_top_zeros.npy"))


def Zb(ts):
    out = np.empty_like(ts)
    for i in range(0, ts.size, 1500): out[i:i + 1500] = R.Z_batch_precise(ts[i:i + 1500])
    return out


def save(name):
    p = os.path.join(OUT, name + ".png"); plt.tight_layout(); plt.savefig(p); plt.close(); print("wrote", p, flush=True)


# ------------------------------------------------------------------ fig1: |Z| arches and floors (E1)
def fig04():
    gam = load("E1"); k0 = 200000
    a, b = gam[k0], gam[k0 + 40]; t = np.linspace(a, b, 6000); z = Zb(t)
    plt.figure(figsize=(7, 2.6)); plt.plot(t - a, np.abs(z), lw=0.8, color="k")
    for k in range(k0, k0 + 41): plt.axvline(gam[k] - a, color="tab:red", lw=0.5, alpha=0.6)
    for k in range(k0, k0 + 40):
        tt = np.linspace(gam[k], gam[k + 1], 200)[1:-1]; zz = Zb(tt); j = np.argmax(np.abs(zz))
        plt.plot(tt[j] - a, abs(zz[j]), "o", ms=3, color="tab:brown")
    plt.xlabel(r"$t-t_0$  ($t_0=%.1f$, ensemble E1)" % a); plt.ylabel(r"$|Z(t)|$")
    plt.title("Arches of |Z| between consecutive zeros (red) and their floors (dots)")
    save("fig04_arches")


# ------------------------------------------------------------------ fig2/3: floors vs CUE
def cue_floors(N, nmat, rng):
    fl = []
    for _ in range(nmat):
        A = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / np.sqrt(2)
        Q, Rm = np.linalg.qr(A); d = np.diag(Rm); Q = Q * (d / np.abs(d))
        ph = np.sort(np.angle(np.linalg.eigvals(Q)))
        ph = np.concatenate([ph, [ph[0] + TWO_PI]])
        for j in range(N):
            x = np.linspace(ph[j], ph[j + 1], 60)[1:-1]
            v = np.exp(np.sum(np.log(np.abs(2 * np.sin((x[:, None] - ph[None, :N]) / 2))), axis=1))
            fl.append(v.max())
    return np.array(fl)


def zeta_floors(w, n=1500, seed=1):
    gam = load(w); rng = np.random.default_rng(seed); ks = np.sort(rng.choice(gam.size - 1, n, replace=False))
    fl = []
    for k in ks:
        tt = np.linspace(gam[k], gam[k + 1], 40)[1:-1]; fl.append(np.abs(Zb(tt)).max())
    return np.array(fl)


def fig08():
    rng = np.random.default_rng(2)
    plt.figure(figsize=(6.5, 3.4)); cols = {"E1": "tab:brown", "E2": "tab:green", "E3": "tab:purple"}
    ratios = {}
    for w, N in (("E1", 12), ("E2", 21), ("E3", 22)):
        zf = zeta_floors(w); q = zf / np.median(zf); qs = np.sort(q)
        plt.plot(qs, np.arange(1, qs.size + 1) / qs.size, color=cols[w], lw=1.2, label=r"$\zeta$, %s ($N\approx%d$)" % (w, N))
        cf = cue_floors(N, 400, rng); qc = np.sort(cf / np.median(cf))
        plt.plot(qc, np.arange(1, qc.size + 1) / qc.size, color=cols[w], lw=1, ls="--", label="CUE(%d)" % N)
        lv = [0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99]
        ratios[w] = np.percentile(q, lv) / np.percentile(qc, lv)
    plt.xlim(0, 3.2); plt.xlabel(r"$q=h_{\rm fl}/\mathrm{med}(h_{\rm fl})$"); plt.ylabel("CDF"); plt.legend(fontsize=7)
    plt.title("Normalised floors: zeta (solid) against CUE (dashed)"); save("fig08_floor_cdf")
    plt.figure(figsize=(5.5, 3.2))
    for w in ("E1", "E2", "E3"): plt.plot([0.5, 1, 2, 5, 10, 25, 50, 75, 90, 99], ratios[w], "o-", color=cols[w], label="%s / CUE" % w)
    plt.xscale("log"); plt.axhline(1, color="k", lw=0.5); plt.xlabel("quantile level (%)"); plt.ylabel(r"quantile ratio $\zeta/\mathrm{CUE}$")
    plt.title("The shallow tail is suppressed arithmetically"); plt.legend(fontsize=7); save("fig09_tail_ratio")


# ------------------------------------------------------------------ fig4: schematic parabola + virtual partners
def fig05():
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    u = np.linspace(-2, 2, 400); h = 0.3; ax[0].plot(u, u * u / 2 - h, "k"); ax[0].axhline(0, color="tab:red", lw=0.6)
    r0 = np.sqrt(2 * h); ax[0].plot([-r0, r0], [0, 0], "o", color="tab:red"); ax[0].plot([0], [-h], "o", color="tab:brown")
    ax[0].set_title(r"tight pair: $Z\approx \frac{Z''}{2}(t-t^*)^2-h_{\rm fl}$"); ax[0].set_xlabel(r"$t-t^*$")
    ax[1].plot([0.5, 0.5], [-0.5, 0.5], color="tab:red", lw=0.6); ax[1].plot([0.5, 0.5], [0.35, -0.35], "o", color="tab:red")
    ax[1].plot([0.5 - 0.33, 0.5 + 0.33], [0, 0], "s", color="tab:brown"); ax[1].annotate("", (0.5 - 0.3, 0.02), (0.5, 0.3), arrowprops=dict(arrowstyle="->", color="gray"))
    ax[1].annotate("", (0.5 + 0.3, 0.02), (0.5, -0.3), arrowprops=dict(arrowstyle="->", color="gray"))
    ax[1].text(0.5 - 0.33, 0.06, r"$s_-=(1/2-y_0)+it^*$", ha="center", fontsize=8); ax[1].text(0.5 + 0.33, 0.06, r"$s_+$", ha="center", fontsize=8)
    ax[1].set_xlim(0.1, 0.9); ax[1].set_ylim(-0.5, 0.5); ax[1].set_xlabel(r"$\sigma$"); ax[1].set_ylabel(r"$t-t^*$"); ax[1].set_title(r"virtual partners, $y_0=\sqrt{2h_{\rm fl}/|Z''|}$")
    save("fig05_schematic")


# ------------------------------------------------------------------ fig5: lambda on a gap, persistence vs dip (synthetic, exact)
def fig01():
    z = np.array([0.0, 1.0, 2.0, 2.6, 3.4, 4.4, 5.2, 6.3]); i = 3; a, b = z[i], z[i + 1]; g = b - a; t0 = a + 0.45 * g
    t = np.linspace(a + 1e-3, b - 1e-3, 2000)
    def lam(h):
        v = np.sum(1 / (t[:, None] - z[None, :]), axis=1); return v + 2 * (t - t0) / ((t - t0) ** 2 + h * h)
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    for axx, h, lab in ((ax[0], 0.30 * g, r"shallow pair, $h=0.30\,g$: dip (3 zeros)"), (ax[1], 0.60 * g, r"deep pair, $h=0.60\,g>g/2$: persists (1 zero)")):
        v = lam(h); axx.plot(t - a, v, "k", lw=1); axx.axhline(0, color="tab:red", lw=0.6); axx.set_ylim(-40, 40)
        nz = int(np.sum(v[:-1] * v[1:] < 0)); axx.set_title(lab + " [%d]" % nz, fontsize=8); axx.set_xlabel(r"$t-\gamma_a$"); axx.set_ylabel(r"$\lambda=\Xi'/\Xi$")
    save("fig01_persistence")


# ------------------------------------------------------------------ fig6: transition table
def fig02():
    t0 = np.array([109.099, 123.602, 133.749, 144.123, 1001.877, 1008.243, 1020.164]); hcl = np.array([2.4, 3.4, 3.6, 3.2, 2.8, 2.6, 3.2])
    plt.figure(figsize=(5.5, 3)); plt.semilogx(t0, hcl, "o", color="tab:brown", label=r"bisection: $h_c\log(t_0/2\pi)$")
    plt.axhspan(2.5, 3.3, color="tab:brown", alpha=0.12, label=r"$2.9\pm0.4$"); plt.axhline(np.pi * 0.966, color="k", ls="--", lw=0.8, label=r"$\pi\cdot\mathrm{med}(g/\bar d)=3.04$")
    plt.xlabel(r"$t_0$"); plt.ylabel(r"$h_c\log(t_0/2\pi)$"); plt.legend(fontsize=7); plt.title("Transition depth and the half-gap threshold"); save("fig02_transition")


# ------------------------------------------------------------------ fig7: realisation margin + y/pi law (E2)
def fig03():
    gam = load("E2"); L = LNTB["E2"] - np.log(TWO_PI); dbar = TWO_PI / L; g = np.diff(gam); r = g / dbar; tm = 0.5 * (gam[:-1] + gam[1:])
    def lam(ts, h=2e-5): return (Zb(ts + h) - Zb(ts - h)) / (2 * h) / Zb(ts) - np.pi / 4 + 9 / (4 * ts)
    rng = np.random.default_rng(5); sel = np.nonzero(r < 0.35)[0]; sel = sel[rng.choice(sel.size, 4000, replace=False)]
    x = (g[sel] / 2) * np.abs(lam(tm[sel])); t0s = rng.uniform(gam[10], gam[-10], 4000); lr = np.abs(lam(t0s)) / L
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.9))
    ax[0].hist(x, bins=60, color="tab:brown", alpha=0.8); ax[0].axvline(1, color="k", ls="--"); ax[0].axvline(0.62, color="gray", ls=":")
    ax[0].set_xlabel(r"$h_0|\lambda_{\rm far}(t_m)|$ at $h_0=g/2$ (tight midpoints, E2)"); ax[0].set_ylabel("count")
    ax[0].set_title("realisation margin: median %.3f, max %.2f" % (np.median(x), x.max()), fontsize=8)
    ys = np.linspace(0.05, 5, 60); P = [np.mean(lr < 1 / y) for y in ys]
    ax[1].plot(ys, P, "o-", ms=2, color="tab:brown", label=r"$P[h_0|\lambda(t_0)|<1]$, random $t_0$"); ax[1].plot(ys, 1 - ys / np.pi, "k--", lw=0.8, label=r"$1-y/\pi$")
    ax[1].set_ylim(0, 1); ax[1].set_xlabel(r"$y=h_0\log(T/2\pi)$"); ax[1].set_ylabel("probability"); ax[1].legend(fontsize=7); ax[1].set_title(r"the $y/\pi$ law", fontsize=8)
    save("fig03_margin_ypi")


# ------------------------------------------------------------------ fig8: evaluator error, naive vs exact-phase (E3)
def fig10():
    from mpmath import mp, mpf, mpc, zeta, siegeltheta, exp as mexp, re as mre
    mp.dps = 25
    def Zmp(t):
        tt = mpf(float(t)); return float(mre(mexp(mpc(0, 1) * siegeltheta(tt)) * zeta(mpc(0.5, tt))))
    def Znaive(ts):
        ts = np.asarray(ts); a = np.sqrt(ts / TWO_PI); N = np.floor(a).astype(int); th = R.theta(ts); s = np.zeros_like(ts)
        for j in range(ts.size):
            n = np.arange(1, N[j] + 1, dtype=float); s[j] = np.sum(np.cos(th[j] - ts[j] * np.log(n)) / np.sqrt(n))
        p = a - N; Psi = np.cos(TWO_PI * (p * p - p - 1 / 16)) / np.cos(TWO_PI * p)
        return 2 * s + np.where(N % 2 == 1, 1.0, -1.0) * (ts / TWO_PI) ** -0.25 * Psi
    gam = load("E3"); rng = np.random.default_rng(9); ks = rng.choice(gam.size - 1, 12, replace=False); T = []
    for k in ks: T += list(np.linspace(gam[k], gam[k + 1], 5)[1:-1])
    T = np.array(T); ref = np.array([Zmp(x) for x in T]); e1 = Znaive(T) - ref; e2 = R.Z_batch_precise(T) - ref
    plt.figure(figsize=(5.5, 3)); plt.semilogy(np.abs(e1), "o", ms=3, label="naive float64 RS (rms %.1e)" % np.sqrt(np.mean(e1 ** 2)))
    plt.semilogy(np.abs(e2), "s", ms=3, label="exact-phase RS (rms %.1e)" % np.sqrt(np.mean(e2 ** 2)))
    plt.xlabel("test point (E3, $t\\approx3.06\\times10^{10}$)"); plt.ylabel("|error| vs mpmath"); plt.legend(fontsize=7); plt.title("Evaluator error"); save("fig10_evaluator")


# ------------------------------------------------------------------ fig07: floor-Euler law (computed)
def fig07():
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31], float); cols = {"E1": "tab:brown", "E2": "tab:green", "E3": "tab:purple"}
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3)); res = {}
    for w in ("E1", "E2", "E3"):
        gam = load(w); L = LNTB[w] - np.log(TWO_PI); g = np.diff(gam); r = g / (TWO_PI / L); idx = np.nonzero(r < 0.35)[0]
        rng = np.random.default_rng(23); sel = np.sort(rng.choice(idx, 1500, replace=False)); ts = np.empty(sel.size); h = np.empty(sel.size)
        for i, k in enumerate(sel):
            a, b = gam[k], gam[k + 1]; T = np.linspace(a, b, 42)[1:-1]; v = Zb(T); j = int(np.argmax(np.abs(v))); ts[i] = T[j]; h[i] = abs(v[j])
        phi = np.log(h / g[sel] ** 2); logP = (np.cos(ts[:, None] * np.log(primes)[None, :]) / np.sqrt(primes)[None, :]).sum(axis=1)
        bp = np.array([2 * np.mean((phi - phi.mean()) * np.cos(ts * np.log(p))) for p in primes]); c = np.sum(bp / np.sqrt(primes)) / np.sum(1 / primes)
        res[w] = (phi, logP, bp, c)
        ax[1].plot(primes, bp, "o", color=cols[w], ms=4, label="%s: $b_p$ (fit $c=%.2f$)" % (w, c)); ax[1].plot(primes, c / np.sqrt(primes), "-", color=cols[w], lw=0.8)
    phi, logP, _, _ = res["E2"]; ax[0].plot(logP, phi, ".", ms=2, color="tab:green", alpha=0.5)
    m, b0 = np.polyfit(logP, phi, 1); xx = np.linspace(logP.min(), logP.max(), 2); ax[0].plot(xx, m * xx + b0, "k-", lw=0.8)
    ax[0].set_xlabel(r"$\log|P_{31}(1/2+it^*)|$"); ax[0].set_ylabel(r"$\varphi=\log(h_{fl}/g^2)$"); ax[0].set_title("E2: corr %.3f, slope %.3f" % (np.corrcoef(logP, phi)[0, 1], m), fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"$p$"); ax[1].set_ylabel(r"$b_p$"); ax[1].set_title(r"per-prime coefficients against $c/\sqrt{p}$", fontsize=8); ax[1].legend(fontsize=6)
    save("fig07_floor_euler")


# ------------------------------------------------------------------ fig10: y0 vs h_thr and FE symmetry (E2)
def fig06():
    from mpmath import mp, mpf, mpc, zeta, pi as mpi
    gam = load("E2"); L = LNTB["E2"] - np.log(TWO_PI); dbar = TWO_PI / L; g = np.diff(gam); r = g / dbar; dens = L / TWO_PI
    idx = np.nonzero(r < 0.35)[0]; rng = np.random.default_rng(3); sel = np.sort(rng.choice(idx, 150, replace=False)); y0s = []; hts = []
    for k in sel:
        a, b = gam[k], gam[k + 1]; T = np.linspace(a, b, 202)[1:-1]; v = Zb(T); j = int(np.argmax(np.abs(v))); ts = T[j]; h = abs(v[j])
        hh = 2e-4 * (b - a); z5 = Zb(np.array([ts - 2 * hh, ts - hh, ts, ts + hh, ts + 2 * hh])); Zpp = (-z5[0] + 16 * z5[1] - 30 * z5[2] + 16 * z5[3] - z5[4]) / (12 * hh * hh)
        lo, hi = max(0, k - 800), min(gam.size, k + 802); others = np.concatenate([gam[lo:k], gam[k + 2:hi]])
        Son = 1 / (ts - a) ** 2 + 1 / (ts - b) ** 2 + np.sum(1 / (ts - others) ** 2) + 2 * dens / min(ts - gam[lo], gam[hi - 1] - ts)
        y0s.append(np.sqrt(2 * h / abs(Zpp))); hts.append(np.sqrt(2 / Son))
    y0s, hts = np.array(y0s), np.array(hts)
    mp.dps = 20; tight = idx[np.argsort(r[idx])[:40]]; lr = []
    for k in tight:
        ts = 0.5 * (gam[k] + gam[k + 1]); y = 0.5 * (gam[k + 1] - gam[k]); tt = mpf(float(ts))
        rm = float(abs(zeta(mpc(mpf(1) / 2 - y, tt)))); rp = float(abs(zeta(mpc(mpf(1) / 2 + y, tt)))); lr.append((np.log(rm) - y * float(mpf.__call__(tt / (2 * mpi)) if False else np.log(ts / TWO_PI)) * 1, np.log(rp)))
    lr = np.array(lr)
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    ax[0].plot(hts, y0s, "o", ms=3, color="tab:green", alpha=0.7); m = max(hts.max(), y0s.max()); ax[0].plot([0, m], [0, m], "k--", lw=0.8)
    ax[0].set_xlabel(r"$h_{\rm thr}=\sqrt{2/S_{\rm on}}$"); ax[0].set_ylabel(r"$y_0=\sqrt{2h_{\rm fl}/|Z''|}$"); ax[0].set_title("median ratio %.4f, corr %.4f" % (np.median(y0s / hts), np.corrcoef(hts, y0s)[0, 1]), fontsize=8)
    ax[1].plot(lr[:, 1], lr[:, 0], "o", ms=3, color="tab:red"); mm = [lr.min(), lr.max()]; ax[1].plot(mm, mm, "k--", lw=0.8)
    ax[1].set_xlabel(r"$\log|\zeta(s_+)|$"); ax[1].set_ylabel(r"$\log|\zeta(s_-)|-y_0\log(t/2\pi)$"); ax[1].set_title("functional-equation symmetry (40 tightest pairs)", fontsize=8)
    save("fig06_identities")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig01", "fig02", "fig03", "fig04", "fig05", "fig06", "fig07", "fig08", "fig10"]
    for w in which:
        t0 = time.time(); globals()[w](); print("  %s done (%.0fs)" % (w, time.time() - t0), flush=True)
