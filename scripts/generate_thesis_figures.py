"""
Genera le 3 figure mancanti per la tesi (PDF vettoriale) SENZA modificare
alcun file esistente del progetto. Output in thesis/figures/.

1. fig_rnd_kde.pdf        - RND vs KDE per una data rappresentativa (Cap 3)
2. fig_mc_drawdown.pdf    - Distribuzione Monte Carlo del max drawdown (Cap 6)
3. fig_sensitivity.pdf    - Heatmap sensitivity profit_target x exit_dte (Cap 7)
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "thesis", "figures")
sys.path.insert(0, PROJ)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})

# ================================================================
# FIGURA 3 (la più semplice): heatmap sensitivity
# ================================================================
def fig_sensitivity():
    df = pd.read_csv(os.path.join(PROJ, "dashboard_data", "sensitivity.csv"))
    piv_sharpe = df.pivot_table(index="exit_dte", columns="profit_target", values="sharpe")
    piv_wr = df.pivot_table(index="exit_dte", columns="profit_target", values="win_rate")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, piv, title, fmt, cmap in [
        (axes[0], piv_sharpe, "Sharpe ratio", "{:.2f}", "RdYlGn"),
        (axes[1], piv_wr, "Win rate (%)", "{:.0f}", "RdYlGn"),
    ]:
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{int(c*100)}%" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([int(i) for i in piv.index])
        ax.set_xlabel("Profit target")
        ax.set_ylabel("Exit DTE")
        ax.set_title(title)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_sensitivity.pdf  griglia:", piv_sharpe.shape,
          " sharpe range:", round(np.nanmin(piv_sharpe.values),2), "-", round(np.nanmax(piv_sharpe.values),2))

# ================================================================
# FIGURA 2: Monte Carlo max drawdown, 3 configurazioni, scenario IS
# Spec dal documento finale: 10.000 percorsi x 200 trade,
# capital-based sizing, WR=65,7%, payoff ratio 2,57,
# magnitudini medie IS: win +1,76%, loss -0,69% dell'equity
# (alla configurazione con vega iniziale ~50 = aggressiva).
# Le percentuali scalano linearmente con risk/buffer di ogni config.
# ================================================================
def fig_mc_drawdown(seed=42):
    rng = np.random.default_rng(seed)
    N_PATHS, N_TRADES = 10_000, 200
    WR = 0.657
    # win/loss medi osservati IS a vega ~54,6 (config aggressiva 3%/1.0x)
    WIN_AGG, LOSS_AGG = 0.0176, 0.0069
    configs = {
        "Conservativa (1%, 2,0x)": (0.01, 2.0, "#2c7fb8"),
        "Moderata (2%, 1,5x)":     (0.02, 1.5, "#41ab5d"),
        "Aggressiva (3%, 1,0x)":   (0.03, 1.0, "#d95f0e"),
    }
    ref = 0.03 / 1.0  # rapporto risk/buffer della config di riferimento

    fig, ax = plt.subplots(figsize=(9, 4.6))
    stats = {}
    for label, (risk, buffer, color) in configs.items():
        scale = (risk / buffer) / ref
        win, loss = WIN_AGG * scale, LOSS_AGG * scale
        # simulazione vettoriale: rendimenti moltiplicativi (capital-based)
        wins = rng.random((N_PATHS, N_TRADES)) < WR
        rets = np.where(wins, win, -loss)
        equity = np.cumprod(1 + rets, axis=1)
        equity = np.hstack([np.ones((N_PATHS, 1)), equity])
        peak = np.maximum.accumulate(equity, axis=1)
        dd = (equity / peak - 1).min(axis=1) * 100  # max DD per percorso, in %
        med, p95 = np.median(dd), np.percentile(dd, 5)
        stats[label] = (med, p95)
        ax.hist(dd, bins=80, alpha=0.55, label=f"{label}  (mediana {med:.0f}%, P95 {p95:.0f}%)",
                color=color, density=True)
        ax.axvline(med, color=color, ls="--", lw=1)
    ax.set_xlabel("Max drawdown del percorso (%)")
    ax.set_ylabel("Densità")
    ax.set_title("Monte Carlo: distribuzione del max drawdown\n(10.000 percorsi × 200 trade, scenario stime IS)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_mc_drawdown.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig_mc_drawdown.pdf")
    for k, (m, p) in stats.items():
        print(f"     {k}: mediana {m:.1f}%  P95 {p:.1f}%   (tabella tesi: cons -7/-11, mod -14/-21, agg -20/-30)")

# ================================================================
# FIGURA 1: RND vs KDE per una data rappresentativa
# Usa la pipeline reale del progetto (src/rnd_engine, src/historical_distribution)
# ================================================================
def fig_rnd_kde():
    from src.data_loader import load_config
    from src.rnd_engine import extract_rnd_for_expiry
    from src.historical_distribution import build_p_distribution

    config = load_config(os.path.join(PROJ, "config", "params.yaml"))

    target_date = "2022-06-15"  # meta del campione IS, mercato in stress
    print("     carico opzioni (solo data target)...")
    chunks = []
    for ch in pd.read_csv(os.path.join(PROJ, "Data", "options_ready_for_backtest.csv"),
                          chunksize=500_000):
        sel = ch[ch["date"] == target_date]
        if len(sel):
            chunks.append(sel)
    df_day = pd.concat(chunks)
    print(f"     {len(df_day)} opzioni il {target_date}")

    # normalizzazione colonne come in src/data_loader.py
    df_day.columns = [c.strip().lower() for c in df_day.columns]
    df_day = df_day.rename(columns={"expiry_date": "expiry"})
    df_day["expiry"] = df_day["expiry"].astype(str).str.strip()
    df_day["option_type"] = df_day["option_type"].str.strip().str.lower()
    df_day["T"] = df_day["dte"] / 365.0
    df_day["moneyness"] = df_day["strike"] / df_day["underlying_price"]

    spot_df = pd.read_csv(os.path.join(PROJ, "Data", "btc_spot_daily.csv"))

    # scegli l'expiry con DTE 30-60 con piu open interest
    cand = (df_day[(df_day["dte"] >= 25) & (df_day["dte"] <= 70)]
            .groupby("expiry")["open_interest"].sum().sort_values(ascending=False))
    print("     expiry candidati:", cand.head(3).to_dict())

    rnd = None
    for expiry in cand.index:
        rnd = extract_rnd_for_expiry(df_day, expiry, config, use_otm_stitching=True)
        if rnd is not None:
            break
    assert rnd is not None, "RND extraction failed on all candidate expiries"

    K, q = rnd["K_valid"], rnd["density"]
    spot, dte = rnd["spot"], rnd["dte"]
    p = build_p_distribution(spot_df, target_date, t_days=int(round(dte)),
                             spot=spot, strikes=K, lookback_days=365)
    assert p is not None, "KDE failed"

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(K / 1000, q * 1000, lw=1.8, color="#1f5fa8",
            label="Densità risk-neutral (RND)")
    ax.plot(K / 1000, p["density_strike"] * 1000, lw=1.8, color="#d95f0e", ls="--",
            label="Densità fisica (KDE, 365 gg)")
    ax.axvline(spot / 1000, color="gray", lw=1, ls=":", label=f"Spot (\\${spot:,.0f})")
    ax.set_xlabel("Strike (migliaia di \\$)")
    ax.set_ylabel("Densità (× 10³)")
    ax.set_title(f"RND vs densità fisica – {target_date}, scadenza {rnd['expiry']} (DTE {dte:.0f})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rnd_kde.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] fig_rnd_kde.pdf  spot={spot:,.0f} dte={dte:.0f} skew={rnd['skewness']:.3f}")

if __name__ == "__main__":
    fig_sensitivity()
    fig_mc_drawdown()
    fig_rnd_kde()
    print("DONE")
