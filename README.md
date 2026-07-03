# Dynamic Asset Allocation under AI-Concentration Risk

### Markowitz vs Michaud Resampling vs Black–Litterman, evaluated through the 2026 AI/semiconductor crisis

**Financial Markets Analytics 2025/2026 — Final Project, Variant B (Dynamic Asset Allocation)**

This project designs, implements and evaluates **out-of-sample** a set of dynamic asset
allocation strategies on a 19-ETF multi-asset universe, with a deliberate focus on the
defining market theme of 2025–2026: the historical concentration of equity markets in
AI/mega-cap technology and its semiconductor supply chain — and what happened when that
concentration was stress-tested by the **2026 AI/semiconductor crisis**.

---

## Research question

> Does resampled (Michaud) or Black–Litterman portfolio construction reduce unwanted
> concentration in AI/tech exposure relative to classical Markowitz, and does that
> translate into better risk-adjusted, out-of-sample performance through the 2026
> AI/semiconductor crisis?

---

## The 2026 AI/semiconductor crisis (the empirical laboratory)

Reconstructed from primary sources in Section 4 of the main notebook:

- **5 Feb 2026 — preview episode.** Hyperscaler capex fears (Amazon guiding to >$200bn of
  2026 capex, Alphabet/Meta nearly doubling theirs); Nasdaq −1.59%.
- **Early June 2026.** Cautious AI-chip outlook from Broadcom; hawkish Fed repricing under
  the new chair (+50bp of hikes priced by December). 5 Jun: Nasdaq 100 −5%, worst day of
  the year.
- **23–26 June 2026 — global chip rout.** Kospi −10% in a session (circuit breaker;
  Samsung and SK Hynix, ≈half the index, both −12%), Micron −13%, Nasdaq −2.2% then ~−4%;
  semiconductor stocks lose **more than $1.3 trillion** of market value — followed by a
  sharp partial rebound on Micron's blowout earnings (25 Jun: Micron +16%, Kospi +5%).

The crisis provides two dated stress windows for evaluation and the economic grounding
for the three Black–Litterman views (AI valuations vs rates, memory-chain weakness,
short-vs-long duration).

---

## Repository structure

```text
.
├── AI_Concentration_Dynamic_Allocation.ipynb   # main notebook (executed, with outputs)
├── 02_sensitivity_analysis.ipynb                # robustness checks (executed)
├── data/cache/                                  # frozen parquet price panel (reproducibility)
├── figures/                                     # exported figures and CSV tables
├── docs/                                        # theory notes
├── archive/                                     # earlier modular prototype (superseded)
├── README.md
└── requirements.txt
```

---

## ETF universe (19 names, 6 buckets)

| Bucket | ETFs |
|---|---|
| Broad equity | SPY, IWM, EFA, EEM |
| **AI/Tech cluster** | **QQQ, SMH, XLK, EWT (Taiwan), EWY (Korea)** |
| Government bonds | SHY, IEF, TLT, TIP |
| Credit | LQD, HYG |
| Real assets | GLD, DBC, VNQ |
| Cash proxy | BIL |

The AI/Tech cluster includes direct exposure to the Taiwanese and Korean semiconductor
supply chain (`EWT`, `EWY`) — the epicentre of the June 2026 rout. Daily prices
2010-07-01 → 2026-06-30 (~16 years, above the 15-year minimum) from Yahoo Finance via
`yfinance`, frozen to parquet after the first download.

---

## Methodology

All strategies share the same **rolling, walk-forward, no-look-ahead backtest engine**:
at each month-end, μ and Σ are estimated on the trailing 60 months (Ledoit–Wolf
shrinkage), weights are optimized (long-only, no leverage, 30% single-asset cap) and held
until the next rebalance. An explicit sanity check verifies that no future information
enters any rebalance.

1. **Strategy A — classical Markowitz**: rolling max-Sharpe (SLSQP).
2. **Strategy B — Michaud resampled frontier**: 300 Monte Carlo re-estimations per
   rebalance, averaged weights (fixed seed).
3. **Strategy C — Black–Litterman** with three crisis-grounded, constant views:
   AI cluster underperforms broad equity (−2%/yr), memory supply chain underperforms US
   mega-cap tech (−3%/yr), short duration outperforms long duration (+1%/yr).
   He–Litterman Ω, τ = 0.03; calibration sensitivity in the companion notebook.
4. **Benchmarks**: 60/40 (SPY/IEF), 1/N across the universe (the "universe index"
   required by the assignment), and a passive 100% QQQ reference.

**Evaluation**: equity curves, annualized volatility, Sharpe (in excess of realized
T-bills), Sortino, max drawdown, Calmar, turnover — plus **concentration diagnostics**
(AI-cluster weight, portfolio HHI) and a **stress-window deep dive** on February and
June 2026 at monthly *and* daily resolution.

**Robustness** (`02_sensitivity_analysis.ipynb`): estimation window 36/60/84m, sample vs
Ledoit–Wolf covariance, weight cap 20%/30%/uncapped, monthly vs quarterly rebalancing,
BL τ × view-magnitude grid, and an explicit AI-bucket budget constraint (≤25%) as the
direct alternative to robust estimation.

---

## Key results

| | Ann. Return | Ann. Vol | Sharpe | Max DD | Jun 2026 | Avg HHI |
|---|---|---|---|---|---|---|
| Markowitz | 7.7% | 7.1% | 0.80 | −17.2% | −4.6% | 0.185 |
| Michaud | 7.0% | 6.4% | 0.78 | −16.7% | −3.8% | 0.123 |
| Black–Litterman | 7.7% | 11.6% | 0.49 | −21.2% | **−0.1%** | 0.195 |
| 60/40 | 8.4% | 9.9% | 0.65 | −21.6% | −0.5% | — |
| 1/N (universe) | 9.3% | 10.8% | 0.68 | −23.4% | −0.2% | 0.053 |
| 100% QQQ | 18.0% | 18.8% | 0.85 | −35.5% | −0.2% | 1.0 |

*(Out-of-sample 2015-08 → 2026-06, monthly rebalancing; Sharpe in excess of realized T-bills.)*

Three findings stand out (full discussion in Section 17 of the main notebook):

1. **Michaud does what it promises**: same Sharpe as classical Markowitz with one-third
   less concentration (HHI 0.123 vs 0.185) and lower turnover.
2. **Black–Litterman was the only construction that actually protected in the crisis**
   (flat in June 2026, −1.8% daily drawdown vs −5.8% for Markowitz), at the cost of the
   worst full-sample Sharpe — constant cautionary views are an insurance premium.
3. **The crisis hit optimized portfolios through an unexpected channel**: at the last
   pre-crisis rebalance Markowitz held only ~14% AI but 41% gold+commodities (trailing
   winners), and the hawkish-Fed shock that triggered the chip rout crashed those too —
   estimation-driven momentum concentration, not the AI theme itself, was the real risk.

---

## Reproducibility

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook   # run AI_Concentration_Dynamic_Allocation.ipynb top to bottom
```

- The first run downloads prices and freezes them to `data/cache/prices_daily.parquet`
  (fixed sample end date 2026-06-30); later runs are fully offline and deterministic.
- Single seed (42) for the Michaud Monte Carlo.
- Every subjective parameter (window, cap, τ, views, …) is set once, in the open, next to
  the markdown cell that justifies it.

---

## References

- Markowitz (1952), *Portfolio Selection*, Journal of Finance
- Michaud (1989), *The Markowitz Optimization Enigma: Is Optimized Optimal?*
- Michaud & Michaud (2008), *Estimation Error and Portfolio Optimization: A Resampling Solution*
- Idzorek (2005), *A Step-by-step Guide to the Black–Litterman Model*
- He & Litterman (1999), *The Intuition Behind Black–Litterman Model Portfolios*
- Ledoit & Wolf (2004), *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*
- Crisis sources: CNN Business, CNBC, NPR, Washington Post, Yahoo Finance, Investing.com
  (full links in Section 4 of the main notebook)

---

## Academic information

**Course:** Financial Markets Analytics 
**Academic year:** 2025/2026
