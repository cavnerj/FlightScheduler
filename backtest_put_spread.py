"""
Backtest: Sell 1-lot 30-day 15Δ/5Δ SPY put spread every trading day
- Bull put spread: sell 15Δ put, buy 5Δ put
- No rolling, no delta hedging
- Same simulated SPY + VIX data as call backtest
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

# ── Black-Scholes helpers ────────────────────────────────────────────────────

def bs_d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_put_price(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_put_delta(S, K, T, r, sigma):
    """Returns put delta (negative number, -1 to 0)."""
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1) - 1.0

def find_put_strike_for_delta(S, T, r, sigma, target_delta):
    """Find put strike for a given target delta (e.g. -0.15 → pass 0.15)."""
    # Strike must be below spot for OTM puts
    lo, hi = S * 0.4, S

    def objective(K):
        return bs_put_delta(S, K, T, r, sigma) + target_delta  # want delta = -target

    try:
        return brentq(objective, lo, hi, xtol=0.01)
    except ValueError:
        return None

# ── Simulate same SPY + VIX data ────────────────────────────────────────────

print("Simulating SPY + VIX data (regime-based GBM, 2022-03-28 → 2026-03-28)...")
np.random.seed(42)

biz_dates = pd.bdate_range(start="2022-03-28", end="2026-03-28")

regimes = [
    ("2023-01-01", -0.18, 0.23, 27.0, 6.0, 0.05),
    ("2024-01-01",  0.24, 0.13, 17.0, 3.5, 0.05),
    ("2025-01-01",  0.22, 0.11, 14.0, 2.5, 0.05),
    ("2025-10-01", -0.09, 0.18, 22.0, 5.0, 0.05),
    ("2026-12-31",  0.04, 0.20, 22.0, 5.0, 0.05),
]

spy_prices = [449.97]
vix_prices = [27.0]
current_regime = 0
vix_now = 27.0
dt = 1 / 252

for i in range(1, len(biz_dates)):
    d = biz_dates[i]
    for j, (end_str, mu, sigma, vix_mu, vix_sig, kappa) in enumerate(regimes):
        if d <= pd.Timestamp(end_str):
            current_regime = j
            break
    _, mu, sigma, vix_mu, vix_sig, kappa = regimes[current_regime]
    z_spy = np.random.randn()
    spy_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_spy
    spy_prices.append(spy_prices[-1] * np.exp(spy_ret))
    z_vix = -0.6 * z_spy + 0.8 * np.random.randn()
    vix_now = vix_now + kappa * (vix_mu - vix_now) + vix_sig * np.sqrt(dt) * z_vix
    vix_now = float(np.clip(vix_now, 10, 80))
    vix_prices.append(vix_now)

data = pd.DataFrame({"SPY": spy_prices, "VIX": vix_prices}, index=biz_dates)
print(f"Data: {data.index[0].date()} → {data.index[-1].date()}  ({len(data)} trading days)\n")

# ── Backtest ─────────────────────────────────────────────────────────────────

DAYS_TO_EXPIRY  = 30
SHORT_DELTA     = 0.15   # sell this put
LONG_DELTA      = 0.05   # buy this put
CONTRACTS       = 1
MULTIPLIER      = 100
r               = 0.045

records = []
dates = data.index.tolist()
date_to_idx = {d: i for i, d in enumerate(dates)}

for i, entry_date in enumerate(dates):
    S     = float(data.loc[entry_date, "SPY"])
    sigma = float(data.loc[entry_date, "VIX"]) / 100.0
    T     = DAYS_TO_EXPIRY / 365.0

    K_short = find_put_strike_for_delta(S, T, r, sigma, SHORT_DELTA)
    K_long  = find_put_strike_for_delta(S, T, r, sigma, LONG_DELTA)
    if K_short is None or K_long is None:
        continue

    prem_short = bs_put_price(S, K_short, T, r, sigma)
    prem_long  = bs_put_price(S, K_long,  T, r, sigma)
    net_credit = prem_short - prem_long
    width      = K_short - K_long
    max_loss   = (width - net_credit) * MULTIPLIER * CONTRACTS

    # Expiry date
    expiry_cal = entry_date + pd.Timedelta(days=DAYS_TO_EXPIRY)
    expiry_idx = None
    for offset in range(0, 10):
        for sign in [1, -1]:
            candidate = expiry_cal + pd.Timedelta(days=offset * sign)
            if candidate in date_to_idx:
                expiry_idx = date_to_idx[candidate]
                break
        if expiry_idx is not None:
            break

    if expiry_idx is None or expiry_idx >= len(dates):
        S_exp = float(data.iloc[-1]["SPY"])
        expiry_date = dates[-1]
        expired = False
    else:
        S_exp = float(data.iloc[expiry_idx]["SPY"])
        expiry_date = dates[expiry_idx]
        expired = True

    # P&L: credit received - spread value at expiry
    short_intrinsic = max(K_short - S_exp, 0.0)
    long_intrinsic  = max(K_long  - S_exp, 0.0)
    spread_value    = short_intrinsic - long_intrinsic
    pnl_per_share   = net_credit - spread_value
    pnl_total       = pnl_per_share * MULTIPLIER * CONTRACTS

    records.append({
        "entry_date":    entry_date.date(),
        "expiry_date":   expiry_date.date(),
        "expired":       expired,
        "S_entry":       round(S, 2),
        "K_short":       round(K_short, 2),
        "K_long":        round(K_long, 2),
        "width":         round(width, 2),
        "IV":            round(sigma * 100, 1),
        "net_credit":    round(net_credit, 4),
        "S_expiry":      round(S_exp, 2),
        "spread_value":  round(spread_value, 4),
        "pnl_per_share": round(pnl_per_share, 4),
        "pnl_total":     round(pnl_total, 2),
        "max_loss":      round(max_loss, 2),
    })

df = pd.DataFrame(records)

# ── Results ──────────────────────────────────────────────────────────────────

total_pnl       = df["pnl_total"].sum()
total_credit    = (df["net_credit"] * MULTIPLIER * CONTRACTS).sum()
win_rate        = (df["pnl_total"] > 0).mean() * 100
avg_pnl         = df["pnl_total"].mean()
max_loss_trade  = df["pnl_total"].min()
max_gain_trade  = df["pnl_total"].max()
max_loss_theory = df["max_loss"].mean()
num_full_loss   = (df["spread_value"] >= df["width"] * 0.95).sum()
df["cum_pnl"]   = df["pnl_total"].cumsum()
max_drawdown    = (df["cum_pnl"].cummax() - df["cum_pnl"]).max()

print("=" * 62)
print("  BACKTEST: Sell 1-lot 30-day 15Δ/5Δ SPY Put Spread (Daily)")
print("  No rolling | No delta hedge | Bull put spread")
print("=" * 62)
print(f"  Period            : {df['entry_date'].iloc[0]} → {df['entry_date'].iloc[-1]}")
print(f"  Total entries     : {len(df):,}")
print(f"  Avg spread width  : ${df['width'].mean():.2f}")
print(f"  Avg net credit    : ${df['net_credit'].mean():.4f}/share  (${df['net_credit'].mean()*100:.2f}/lot)")
print(f"  Avg max loss/lot  : ${max_loss_theory:.2f}")
print(f"  Full-width losses : {num_full_loss:,}  ({num_full_loss/len(df)*100:.1f}%)")
print("-" * 62)
print(f"  Total P&L         : ${total_pnl:>12,.2f}")
print(f"  Total credit in   : ${total_credit:>12,.2f}")
print(f"  Win rate          : {win_rate:.1f}%")
print(f"  Avg P&L/trade     : ${avg_pnl:>10,.2f}")
print(f"  Best trade        : ${max_gain_trade:>10,.2f}")
print(f"  Worst trade       : ${max_loss_trade:>10,.2f}")
print(f"  Max drawdown      : ${max_drawdown:>10,.2f}")
print("=" * 62)

df["year"] = pd.to_datetime(df["entry_date"]).dt.year
yearly = df.groupby("year")["pnl_total"].agg(["sum", "mean", "count"])
yearly.columns = ["Total P&L", "Avg/trade", "Trades"]
yearly["Total P&L"] = yearly["Total P&L"].map("${:,.2f}".format)
yearly["Avg/trade"] = yearly["Avg/trade"].map("${:,.2f}".format)
print("\nYearly Breakdown:")
print(yearly.to_string())
print()

df.to_csv("/home/user/FlightScheduler/backtest_put_spread_results.csv", index=False)
print("Full trade log saved to backtest_put_spread_results.csv")

# ── Chart ────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(pd.to_datetime(df["entry_date"]), df["cum_pnl"], color="darkorange", linewidth=1.2)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.fill_between(pd.to_datetime(df["entry_date"]), df["cum_pnl"], 0,
                where=df["cum_pnl"] >= 0, alpha=0.15, color="green")
ax.fill_between(pd.to_datetime(df["entry_date"]), df["cum_pnl"], 0,
                where=df["cum_pnl"] < 0, alpha=0.15, color="red")
ax.set_title("Cumulative P&L — Sell 1-lot 30-day 15\u03945\u0394 SPY Put Spread Daily\n"
             "(No Roll | No Delta Hedge | 2022\u20132026)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative P&L")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("/home/user/FlightScheduler/cumulative_pnl_put_spread.png", dpi=150)
print("Chart saved to cumulative_pnl_put_spread.png")
