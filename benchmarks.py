"""
benchmarks.py
=============
Benchmark "naive", non ottimizzati: equal-weight (1/N) e 60/40.

Non stimano nulla (nessun mu, nessuna Sigma): i pesi TARGET sono fissi e
noti a priori (da config.py), quindi per costruzione non c'e' nessun rischio
di look-ahead nella scelta dei pesi.

L'unica cosa che replica la logica del motore walk-forward e' il
RIBILANCIAMENTO: alle stesse date di backtest.py (stesso ESTIMATION_WINDOW e
REBALANCE_EVERY, per un confronto out-of-sample equo tra strategie attive e
benchmark), i pesi vengono riportati al target, con lo stesso trattamento di
costo di transazione sul turnover generato dal drift infra-periodo. Per
questo il modulo riusa direttamente le funzioni di backtest.py (turnover,
transaction_cost, _simulate_period, rebalance_indices) invece di duplicarle.

Nota: i benchmark non sono soggetti ai vincoli "attivi" dell'ottimizzazione
(config.LONG_ONLY, WEIGHT_CAP, CLASS_BUDGET) -- sono per definizione regole
fisse e trasparenti (1/N e 60/40 equipesato per sleeve), non il risultato di
un problema di ottimo vincolato.
"""

from __future__ import annotations
import pandas as pd

import config as cfg
import backtest as bt


# ---------------------------------------------------------------------------
# 1. Pesi target dei benchmark
# ---------------------------------------------------------------------------
def equal_weight_target(assets: list[str]) -> pd.Series:
    """Portafoglio 1/N: equipesato su tutto l'universo disponibile."""
    n = len(assets)
    return pd.Series(1.0 / n, index=assets, name="equal_weight")


def sixty_forty_target(assets: list[str]) -> pd.Series:
    """Portafoglio 60/40: 60% equity, 40% bond (config.SIXTY_FORTY_SPLIT),
    equipesato all'interno di ciascuna "sleeve" (config.SIXTY_FORTY_EQUITY /
    SIXTY_FORTY_BOND).

    Se qualche ticker di una sleeve non e' presente in `assets` (es. scartato
    da clean_prices per copertura insufficiente), il peso della sleeve viene
    redistribuito equamente sui soli ticker superstiti di quella sleeve,
    mantenendo lo split 60/40 complessivo. Solleva un errore se una sleeve
    resta completamente vuota (lo split non sarebbe piu' definibile).
    """
    equity_w, bond_w = cfg.SIXTY_FORTY_SPLIT
    equity_assets = [t for t in cfg.SIXTY_FORTY_EQUITY if t in assets]
    bond_assets = [t for t in cfg.SIXTY_FORTY_BOND if t in assets]

    if not equity_assets or not bond_assets:
        raise ValueError(
            "sixty_forty_target: sleeve equity o bond vuota rispetto agli "
            "asset disponibili; impossibile costruire il 60/40."
        )

    w = pd.Series(0.0, index=assets, name="sixty_forty")
    w[equity_assets] = equity_w / len(equity_assets)
    w[bond_assets] = bond_w / len(bond_assets)
    return w


def target_weights(assets: list[str], method: str) -> pd.Series:
    """Dispatcher per i pesi target dei benchmark."""
    if method == "equal_weight":
        return equal_weight_target(assets)
    if method == "sixty_forty":
        return sixty_forty_target(assets)
    raise ValueError(f"target_weights: benchmark '{method}' non riconosciuto.")


# ---------------------------------------------------------------------------
# 2. Motore di ribilanciamento per i benchmark (riusa backtest.py)
# ---------------------------------------------------------------------------
def run_benchmark(
    returns: pd.DataFrame,
    method: str,
    charge_initial_cost: bool = False,
    verbose: bool = False,
) -> dict:
    """Esegue il benchmark walk-forward: STESSE date di ribilanciamento del
    motore ottimizzato (backtest.rebalance_indices), STESSO trattamento di
    turnover/costi. I pesi target sono pero' costanti nel tempo (non
    ristimati ad ogni data), quindi il turnover ad ogni ribilanciamento e'
    generato solo dal drift dei prezzi tra un ribilanciamento e l'altro, non
    da un cambio di view.

    Ritorna lo stesso formato di backtest.run_backtest, cosi' da poter
    passare indifferentemente strategie attive e benchmark a metrics.py:
      - "returns"  : Serie dei rendimenti netti mensili di portafoglio
      - "weights"  : DataFrame dei pesi TARGET ad ogni data di ribilanciamento
      - "turnover" : Serie del turnover ad ogni ribilanciamento
      - "costs"    : Serie dei costi di transazione ad ogni ribilanciamento
    """
    n_obs = len(returns)
    rebal_idx = bt.rebalance_indices(n_obs)
    dates = returns.index
    assets = list(returns.columns)

    w_target = target_weights(assets, method)

    all_port_returns = []
    weights_history = {}
    turnover_history = {}
    cost_history = {}

    w_prior_drifted = None

    for k, i in enumerate(rebal_idx):
        is_first = (k == 0)
        turn = bt.turnover(w_target, w_prior_drifted)
        cost = bt.transaction_cost(turn) if (not is_first or charge_initial_cost) else 0.0

        weights_history[dates[i]] = w_target
        turnover_history[dates[i]] = turn
        cost_history[dates[i]] = cost

        j_end = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else n_obs
        period_returns = returns.iloc[i:j_end]

        port_returns, w_prior_drifted = bt._simulate_period(w_target, period_returns, cost)
        all_port_returns.append(port_returns)

        if verbose:
            print(f"[{method}] ribilanciamento {dates[i].date()}: "
                  f"turnover={turn:.3f}, costo={cost*10_000:.1f}bps")

    port_returns_full = pd.concat(all_port_returns).sort_index()
    weights_df = pd.DataFrame(weights_history).T
    weights_df.index.name = "date"
    turnover_s = pd.Series(turnover_history, name="turnover")
    cost_s = pd.Series(cost_history, name="cost")

    return {
        "returns": port_returns_full,
        "weights": weights_df,
        "turnover": turnover_s,
        "costs": cost_s,
    }


# ---------------------------------------------------------------------------
# 3. Tutti i benchmark, sullo stesso periodo out-of-sample
# ---------------------------------------------------------------------------
def run_all_benchmarks(
    returns: pd.DataFrame,
    benchmarks: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, dict]:
    """Esegue run_benchmark per ogni benchmark in `benchmarks` (default:
    config.BENCHMARKS = ["equal_weight", "sixty_forty"])."""
    benchmarks = benchmarks or cfg.BENCHMARKS
    results = {}
    for method in benchmarks:
        if verbose:
            print(f"\n--- Benchmark: {method} ---")
        results[method] = run_benchmark(returns, method=method, verbose=verbose)
    return results


if __name__ == "__main__":
    # Piccolo self-test manuale: usa la cache di data_loader se presente.
    from data_loader import load_data

    data = load_data(use_cache=True)
    returns = data["returns"]

    results = run_all_benchmarks(returns, verbose=True)
    combined = bt.combine_returns(results)

    print("\n=== Rendimenti netti mensili benchmark (testa) ===")
    print(combined.head())
    print("\n=== Turnover medio per benchmark ===")
    for name, res in results.items():
        print(f"  {name}: {res['turnover'].mean():.3f}")
