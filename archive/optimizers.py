"""
optimizers.py
=============
Ottimizzatori di portafoglio: minimum-variance, max-Sharpe (Markowitz
classico via scipy) e la loro versione "resampled" alla Michaud.

Responsabilita' di questo modulo (e solo queste):
  1. tradurre i vincoli soggettivi di config.py (long-only, weight cap,
     budget di asset class, fully invested) in bounds/constraints scipy;
  2. risolvere min-variance e max-Sharpe dati mu, Sigma (stimati altrove,
     da estimators.py, su una finestra passata -> nessun look-ahead qui);
  3. implementare il resampling di Michaud: simulare M portafogli (mu_m,
     Sigma_m) attorno alla stima puntuale (mu, Sigma) e mediare i pesi
     ottimi -> pesi meno sensibili al rumore di stima.

Formule di riferimento (vedi PDF del corso):
  min-variance:  min  w' Sigma w        s.t. w' 1 = 1  (e vincoli)
  max-Sharpe:    max (w' mu - rf) / sqrt(w' Sigma w)    s.t. w' 1 = 1 (e vincoli)
  Michaud:       w* = (1/M) * sum_m w_m ,  w_m ottimo su (mu_m, Sigma_m) simulati

Questo modulo NON scarica ne' stima nulla da solo: riceve sempre mu/Sigma
(o la finestra di rendimenti, per Michaud) gia' pronti da estimators.py.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config as cfg
import estimators as est


# ---------------------------------------------------------------------------
# 1. Vincoli: bounds e constraints scipy a partire da config.py
# ---------------------------------------------------------------------------
def _bounds(assets: list[str]) -> list[tuple[float, float]]:
    """Bounds per singolo peso, da config.LONG_ONLY e config.WEIGHT_CAP.

    LONG_ONLY=True  -> lower = 0 (no short)
    LONG_ONLY=False -> lower = -cap (short ammesso fino al cap)
    WEIGHT_CAP=None -> nessun cap superiore (solo il vincolo fully-invested
                        e i bounds impliciti [-1,1]/[0,1] lo limitano comunque)
    """
    cap = cfg.WEIGHT_CAP if cfg.WEIGHT_CAP is not None else 1.0
    lower = 0.0 if cfg.LONG_ONLY else -cap
    upper = cap
    return [(lower, upper)] * len(assets)


def _budget_constraints(assets: list[str]) -> list[dict]:
    """Vincoli di budget per asset class, da config.CLASS_BUDGET.

    Per ogni classe con un budget (lo, hi) definito, aggiunge due vincoli di
    disuguaglianza scipy (fun >= 0):
      sum(w_classe) - lo >= 0
      hi - sum(w_classe) >= 0

    Se un asset della finestra corrente non compare in config.ASSET_CLASS
    (non dovrebbe succedere, ma per sicurezza) viene ignorato nel budget.
    Se config.CLASS_BUDGET e' None, ritorna lista vuota (nessun vincolo).
    """
    if cfg.CLASS_BUDGET is None:
        return []

    cons = []
    for cls, (lo, hi) in cfg.CLASS_BUDGET.items():
        idx = [i for i, t in enumerate(assets) if cfg.ASSET_CLASS.get(t) == cls]
        if not idx:
            continue
        idx = np.array(idx)
        cons.append({"type": "ineq", "fun": lambda w, idx=idx, lo=lo: w[idx].sum() - lo})
        cons.append({"type": "ineq", "fun": lambda w, idx=idx, hi=hi: hi - w[idx].sum()})
    return cons


def _constraints(assets: list[str]) -> list[dict]:
    """Tutti i vincoli di uguaglianza/disuguaglianza (fully-invested + budget)."""
    cons = []
    if cfg.FULLY_INVESTED:
        cons.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    cons.extend(_budget_constraints(assets))
    return cons


def _initial_guess(n_assets: int) -> np.ndarray:
    """Punto di partenza per l'ottimizzatore: equal-weight (sempre feasible
    rispetto a fully-invested + long-only; ragionevole anche con budget e
    cap moderati come quelli di config.py)."""
    return np.repeat(1.0 / n_assets, n_assets)


# ---------------------------------------------------------------------------
# 2. Minimum variance
# ---------------------------------------------------------------------------
def min_variance(sigma: pd.DataFrame) -> pd.Series:
    """Portafoglio a varianza minima: min w' Sigma w, s.t. vincoli di config.

    Non usa mu (e' proprio il suo punto di forza: nessuna stima rumorosa dei
    rendimenti attesi entra nel problema).
    """
    assets = list(sigma.columns)
    n = len(assets)
    S = sigma.values

    def objective(w):
        return w @ S @ w

    result = minimize(
        objective,
        x0=_initial_guess(n),
        method="SLSQP",
        bounds=_bounds(assets),
        constraints=_constraints(assets),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        warnings.warn(f"min_variance: ottimizzazione non convergente ({result.message}).")

    return pd.Series(result.x, index=assets, name="min_variance")


# ---------------------------------------------------------------------------
# 3. Max Sharpe
# ---------------------------------------------------------------------------
def max_sharpe(mu: pd.Series, sigma: pd.DataFrame, rf: float = 0.0) -> pd.Series:
    """Portafoglio a massimo Sharpe: max (w' mu - rf) / sqrt(w' Sigma w).

    rf e' il risk-free MENSILE (stessa unita' di mu, coerente con
    data_loader.load_riskfree). Risolto minimizzando lo Sharpe con segno
    invertito via SLSQP; la non-convessita' del rapporto e' gestita bene in
    pratica con bounds/cap moderati come in config.py e un punto di partenza
    equal-weight.
    """
    assets = list(mu.index)
    n = len(assets)
    m = mu.values
    S = sigma.loc[assets, assets].values

    def neg_sharpe(w):
        port_ret = w @ m - rf
        port_vol = np.sqrt(w @ S @ w)
        if port_vol < 1e-12:
            return 0.0  # portafoglio degenere (quasi certo/varianza nulla): non spingere l'ottimizzatore li'
        return -port_ret / port_vol

    result = minimize(
        neg_sharpe,
        x0=_initial_guess(n),
        method="SLSQP",
        bounds=_bounds(assets),
        constraints=_constraints(assets),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        warnings.warn(f"max_sharpe: ottimizzazione non convergente ({result.message}).")

    return pd.Series(result.x, index=assets, name="max_sharpe")


# ---------------------------------------------------------------------------
# 4. Dispatcher comune
# ---------------------------------------------------------------------------
def optimize(mu: pd.Series, sigma: pd.DataFrame, method: str, rf: float = 0.0) -> pd.Series:
    """Dispatcher: risolve il portafoglio ottimo secondo `method`.

    method in {"min_variance", "max_sharpe"}. Usato sia direttamente dal
    motore walk-forward, sia internamente da michaud_resampled() ad ogni
    path Monte Carlo simulato.
    """
    if method == "min_variance":
        return min_variance(sigma)
    if method == "max_sharpe":
        return max_sharpe(mu, sigma, rf=rf)
    raise ValueError(f"optimize: metodo '{method}' non riconosciuto.")


# ---------------------------------------------------------------------------
# 5. Resampling di Michaud
# ---------------------------------------------------------------------------
def michaud_resampled(
    returns_window: pd.DataFrame,
    method: str = "max_sharpe",
    rf: float = 0.0,
    n_resamples: int | None = None,
    resample_length: int | None = None,
    seed: int | None = None,
    mu_method: str | None = None,
    cov_method: str | None = None,
) -> pd.Series:
    """Portafoglio resampled alla Michaud (Michaud & Michaud, 2008).

    Procedura:
      1. stima puntuale (mu, Sigma) sulla finestra passata, con lo stesso
         stimatore usato altrove nel progetto (estimators.estimate);
      2. simula M = n_resamples serie storiche sintetiche di lunghezza
         resample_length da una Normale multivariata (mu, Sigma) -- la
         convenzione di Michaud e' resample_length = estimation window,
         cosi' l'incertezza simulata replica quella della stima reale
         (cfr. config.RESAMPLE_LENGTH = config.ESTIMATION_WINDOW);
      3. su ciascuna serie simulata ri-stima (mu_m, Sigma_m) e risolve
         l'ottimo w_m con lo stesso `method` (min-variance o max-Sharpe);
      4. media i pesi: w* = (1/M) * sum_m w_m.

    Il risultato e' un portafoglio meno sensibile al rumore di stima di
    mu/Sigma rispetto all'ottimo "puntuale" (che spesso concentra il peso su
    pochi asset a causa dell'error-maximizing di Markowitz).

    Nessun look-ahead: la simulazione parte solo da (mu, Sigma) stimati sulla
    finestra passata `returns_window`, non dai rendimenti futuri.
    """
    n_resamples = n_resamples or cfg.N_RESAMPLES
    resample_length = resample_length or cfg.RESAMPLE_LENGTH
    seed = cfg.SEED if seed is None else seed

    mu0, sigma0 = est.estimate(returns_window, mu_method=mu_method, cov_method=cov_method)
    assets = list(returns_window.columns)
    rng = np.random.default_rng(seed)

    weights_sum = pd.Series(0.0, index=assets)
    n_failed = 0

    for _ in range(n_resamples):
        sim_values = rng.multivariate_normal(mu0.values, sigma0.values, size=resample_length)
        sim_returns = pd.DataFrame(sim_values, columns=assets)

        try:
            mu_m, sigma_m = est.estimate(sim_returns, mu_method=mu_method, cov_method=cov_method)
            w_m = optimize(mu_m, sigma_m, method=method, rf=rf)
        except Exception:
            n_failed += 1
            continue

        weights_sum = weights_sum.add(w_m, fill_value=0.0)

    n_ok = n_resamples - n_failed
    if n_ok == 0:
        raise RuntimeError("michaud_resampled: tutti i path Monte Carlo sono falliti.")
    if n_failed > 0:
        warnings.warn(f"michaud_resampled: {n_failed}/{n_resamples} path falliti e scartati.")

    w_star = weights_sum / n_ok
    w_star.name = "michaud_resampled"
    return w_star


# ---------------------------------------------------------------------------
# 6. Convenienza: tutte le strategie attive su una finestra
# ---------------------------------------------------------------------------
def run_strategies(
    returns_window: pd.DataFrame,
    strategies: list[str] | None = None,
    rf: float = 0.0,
) -> dict[str, pd.Series]:
    """Calcola i pesi di tutte le strategie in `strategies` (default:
    config.STRATEGIES) sulla stessa finestra di rendimenti passati.

    mu e Sigma vengono stimati UNA SOLA VOLTA (per min_variance/max_sharpe);
    michaud_resampled ristima internamente ad ogni path Monte Carlo, quindi
    riceve direttamente `returns_window`.
    """
    strategies = strategies or cfg.STRATEGIES
    mu, sigma = est.estimate(returns_window)

    out = {}
    for strat in strategies:
        if strat == "michaud_resampled":
            out[strat] = michaud_resampled(returns_window, method="max_sharpe", rf=rf)
        else:
            out[strat] = optimize(mu, sigma, method=strat, rf=rf)
    return out


if __name__ == "__main__":
    # Piccolo self-test manuale: usa la cache di data_loader se presente.
    from data_loader import load_data

    data = load_data(use_cache=True)
    returns = data["returns"]
    rf_last = float(data["riskfree"].iloc[-1])

    window = returns.iloc[-cfg.ESTIMATION_WINDOW:]

    print("=== Min-variance ===")
    print(min_variance(est.estimate(window)[1]).round(4).sort_values(ascending=False))

    print("\n=== Max-Sharpe ===")
    mu, sigma = est.estimate(window)
    print(max_sharpe(mu, sigma, rf=rf_last).round(4).sort_values(ascending=False))

    print(f"\n=== Michaud resampled (max_sharpe, M={cfg.N_RESAMPLES}) ===")
    w_mich = michaud_resampled(window, method="max_sharpe", rf=rf_last)
    print(w_mich.round(4).sort_values(ascending=False))
