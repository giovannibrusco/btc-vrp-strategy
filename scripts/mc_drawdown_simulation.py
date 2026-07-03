"""
Monte Carlo definitivo per il Capitolo 6 della tesi.
Metodo: circular block bootstrap dei rendimenti giornalieri MTM reali (IS),
10.000 percorsi, orizzonte 845 giorni, blocco 20 giorni, seed 2026.
Stress: perdite giornaliere amplificate x1.5.
Output: numeri tabella + figura thesis/figures/fig_mc_drawdown.pdf
"""
import numpy as np, pandas as pd, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = "/Users/giovannibrusco/Desktop/Progetto BTC"
OUT = os.path.join(PROJ, "thesis", "figures")
rng = np.random.default_rng(2026)
N, BLOCK = 10_000, 20
RUIN_FRAC = 0.10  # $10k su $100k

def load_daily(c):
    df = pd.read_csv(f"{PROJ}/dashboard_data/daily_equity_{c}_is.csv")
    eqcol = [x for x in df.columns if "equity" in x.lower()][0]
    return df[eqcol].values.astype(float)

def block_paths(r, horizon):
    n = len(r); nb = int(np.ceil(horizon / BLOCK))
    starts = rng.integers(0, n, size=(N, nb)); offs = np.arange(BLOCK)
    idx = (starts[:, :, None] + offs[None, None, :]) % n
    return r[idx.reshape(N, -1)[:, :horizon]]

def dd_and_ruin(rets):
    eq = np.cumprod(1 + rets, axis=1)
    eq = np.hstack([np.ones((N, 1)), eq])
    peak = np.maximum.accumulate(eq, axis=1)
    dd = (eq / peak - 1).min(axis=1) * 100
    ruined = (eq.min(axis=1) <= RUIN_FRAC).mean() * 100
    return dd, ruined

CONFIGS = [("cons", "Conservativa (1\\%)", "#2c7fb8"),
           ("mod", "Moderata (2\\%)", "#41ab5d"),
           ("agg", "Aggressiva (3\\%)", "#d95f0e")]

results = {}
H = None
for c, label, color in CONFIGS:
    eq = load_daily(c)
    r = np.diff(eq) / eq[:-1]
    if H is None: H = len(r)
    realized = ((eq / np.maximum.accumulate(eq)) - 1).min() * 100

    base = block_paths(r, H)
    dd_b, ruin_b = dd_and_ruin(base)

    r_stress = np.where(r < 0, r * 1.5, r)
    stress = block_paths(r_stress, H)
    dd_s, ruin_s = dd_and_ruin(stress)

    results[c] = dict(realized=realized,
                      med=np.median(dd_b), p95=np.percentile(dd_b, 5), p99=np.percentile(dd_b, 1),
                      ruin=ruin_b,
                      s_med=np.median(dd_s), s_p95=np.percentile(dd_s, 5), ruin_s=ruin_s,
                      dd_dist=dd_b)

print(f"horizon={H} giorni, N={N}, block={BLOCK}, seed=2026")
print(f"{'config':>6} | {'realized':>8} | {'med':>7} {'P95':>7} {'P99':>7} | {'stress med':>10} {'stress P95':>10} | {'P(ruin)':>8} {'P(ruin)s':>8}")
for c, label, _ in CONFIGS:
    v = results[c]
    print(f"{c:>6} | {v['realized']:7.1f}% | {v['med']:6.1f}% {v['p95']:6.1f}% {v['p99']:6.1f}% | "
          f"{v['s_med']:9.1f}% {v['s_p95']:9.1f}% | {v['ruin']:7.2f}% {v['ruin_s']:7.2f}%")

# ── Figura: distribuzione del max drawdown per configurazione ──
fig, ax = plt.subplots(figsize=(9, 4.6))
for c, label, color in CONFIGS:
    v = results[c]
    lab = label.replace("\\%", "%")
    ax.hist(v["dd_dist"], bins=90, alpha=0.55, density=True, color=color,
            label=f"{lab}: mediana {v['med']:.0f}%, P95 {v['p95']:.0f}%")
    ax.axvline(v["med"], color=color, ls="--", lw=1)
    ax.axvline(v["realized"], color=color, ls=":", lw=1.5)
ax.set_xlabel("Max drawdown del percorso (%)")
ax.set_ylabel("Densità")
ax.set_title("Monte Carlo: distribuzione del max drawdown\n"
             "(block bootstrap dei rendimenti giornalieri MTM in-sample, 10.000 percorsi)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_mc_drawdown.pdf"), bbox_inches="tight")
print("figura salvata: thesis/figures/fig_mc_drawdown.pdf")


# ── Figura 2: percorsi di equity simulati (fan chart) ───────────────
def fig_equity_paths():
    """Genera thesis/figures/fig_mc_paths.pdf: campione di percorsi,
    banda 5-95 pct, mediana, curva realizzata. Asse Y stretto sui dati
    (percentili 1-99); la soglia di rovina (10% eq.) e' fuori scala."""
    rng2 = np.random.default_rng(2026)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0), sharex=True)
    handles = None
    for ax, (c, label, color) in zip(axes, CONFIGS):
        lab = label.replace("\\%", "%")
        eq_real = load_daily(c)
        r = np.diff(eq_real) / eq_real[:-1]
        H = len(r)
        n = len(r); nb = int(np.ceil(H / BLOCK))
        starts = rng2.integers(0, n, size=(N, nb)); offs = np.arange(BLOCK)
        idx = (starts[:, :, None] + offs[None, None, :]) % n
        rets = r[idx.reshape(N, -1)[:, :H]]
        eq = np.cumprod(1 + rets, axis=1)
        eq = np.hstack([np.ones((N, 1)), eq]) * 100
        x = np.arange(H + 1)
        real = eq_real / eq_real[0] * 100
        for i in rng2.choice(N, 120, replace=False):
            ax.plot(x, eq[i], color=color, alpha=0.05, lw=0.5, zorder=1)
        p5, p50, p95 = np.percentile(eq, [5, 50, 95], axis=0)
        h1 = ax.fill_between(x, p5, p95, color=color, alpha=0.18, zorder=2, label="banda 5°-95° pct")
        h2, = ax.plot(x, p50, color=color, lw=1.7, zorder=3, label="mediana simulata")
        h3, = ax.plot(np.arange(len(real)), real, color="black", lw=1.4, ls="--",
                      zorder=4, label="backtest realizzato")
        lo = min(np.percentile(eq, 1, axis=0).min(), real.min())
        hi = max(np.percentile(eq, 99, axis=0).max(), real.max())
        pad = (hi - lo) * 0.04
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("Giorni")
        ax.grid(alpha=0.25)
        if handles is None:
            handles = [h1, h2, h3]
    axes[0].set_ylabel("Equity (base 100 = 100.000 USD)")
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Monte Carlo: percorsi simulati di equity (block bootstrap, 10.000 percorsi, campione di 120 mostrato)", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(OUT, "fig_mc_paths.pdf"), bbox_inches="tight")
    print("figura salvata: thesis/figures/fig_mc_paths.pdf")

fig_equity_paths()
