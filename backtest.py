"""
backtest.py
===========
Motore walk-forward: il cuore del progetto.

Ad ogni data di ribilanciamento (ogni cfg.REBALANCE_EVERY mesi):
  1. costruisce la finestra di stima (rolling o expanding, cfg.WINDOW_TYPE)
     usando SOLO rendimenti gia' realizzati (fino al mese precedente incluso);
  2. stima mu/Sigma (estimators.py) e risolve i pesi ottimi per la strategia
     scelta (optimizers.py);
  3. applica quei pesi ai rendimenti REALIZZATI dei mesi successivi, fino al
     ribilanciamento seguente, facendo "driftare" i pesi mese per mese;
  4. addebita i costi di transazione (cfg.TRANSACTION_COST_BPS) sul turnover
     registrato ad ogni ribilanciamento.

Principio anti-look-ahead (il piu' importante di tutto il progetto): la
finestra di stima usata per decidere i pesi al ribilanciamento in posizione i
include AL MASSIMO i rendimenti fino alla posizione i-1. I pesi cosi' decisi
vengono applicati SOLO ai rendimenti dalla posizione i in avanti (out of
sample rispetto alla stima). Questo modulo non calcola mai nulla usando
rendimenti futuri rispetto alla data di decisione.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import config as cfg
import estimators as estim
import optimizers as opt


# ---------------------------------------------------------------------------
# 1. Date di ribilanciamento e finestre di stima
# ---------------------------------------------------------------------------
def _first_rebalance_idx(n_obs: int) -> int:
    """Indice (posizionale) del primo ribilanciamento: appena disponibile
    una finestra di stima completa di cfg.ESTIMATION_WINDOW mesi.

    Vale sia per finestra rolling sia expanding: in entrambi i casi serve
    ALMENO ESTIMATION_WINDOW mesi di storia prima del primo ribilanciamento,
    altrimenti la stima di Sigma userebbe troppo poche osservazioni (cfr. il
    controllo di sanita' in estimators.estimate).
    """
    if n_obs <= cfg.ESTIMATION_WINDOW:
        raise ValueError(
            f"backtest: solo {n_obs} mesi di rendimenti disponibili, "
            f"servono almeno {cfg.ESTIMATION_WINDOW} (ESTIMATION_WINDOW) "
            f"per la prima finestra di stima."
        )
    return cfg.ESTIMATION_WINDOW


def rebalance_indices(n_obs: int) -> list[int]:
    """Indici posizionali (in returns.index) di tutte le date di
    ribilanciamento, ogni cfg.REBALANCE_EVERY mesi a partire dal primo
    disponibile."""
    i0 = _first_rebalance_idx(n_obs)
    return list(range(i0, n_obs, cfg.REBALANCE_EVERY))


def estimation_window(returns: pd.DataFrame, i: int) -> pd.DataFrame:
    """Finestra di stima da usare per decidere i pesi al ribilanciamento in
    posizione i: usa SOLO returns.iloc[:i] (mesi 0..i-1).

    WINDOW_TYPE = "rolling"   -> ultimi ESTIMATION_WINDOW mesi (finestra
                                  mobile a lunghezza fissa, piu' reattiva).
    WINDOW_TYPE = "expanding" -> tutta la storia disponibile fino a i-1
                                  (finestra ancorata, piu' stabile ma con
                                  memoria "infinita" del passato lontano).
    """
    if cfg.WINDOW_TYPE == "rolling":
        start = max(0, i - cfg.ESTIMATION_WINDOW)
        return returns.iloc[start:i]
    if cfg.WINDOW_TYPE == "expanding":
        return returns.iloc[:i]
    raise ValueError(f"estimation_window: WINDOW_TYPE '{cfg.WINDOW_TYPE}' non riconosciuto.")


# ---------------------------------------------------------------------------
# 2. Turnover e costi di transazione
# ---------------------------------------------------------------------------
def turnover(w_target: pd.Series, w_prior: pd.Series | None) -> float:
    """Turnover L1 tra i pesi target e i pesi "drifted" prima del
    ribilanciamento: sum(|w_target - w_prior|).

    w_prior=None -> primo ribilanciamento, si parte da cash (pesi tutti a
    zero): il turnover coincide con la somma dei pesi target (di solito 1,
    se FULLY_INVESTED), cioe' l'investimento iniziale conterebbe come
    turnover pieno (vedi charge_initial_cost in run_backtest).
    """
    if w_prior is None:
        w_prior = pd.Series(0.0, index=w_target.index)
    else:
        w_prior = w_prior.reindex(w_target.index).fillna(0.0)
    return float((w_target - w_prior).abs().sum())


def transaction_cost(turn: float, bps: float | None = None) -> float:
    """Costo proporzionale al turnover, in frazione di portafoglio."""
    bps = cfg.TRANSACTION_COST_BPS if bps is None else bps
    return turn * bps / 10_000.0


# ---------------------------------------------------------------------------
# 3. Simulazione di un singolo periodo (tra due ribilanciamenti)
# ---------------------------------------------------------------------------
def _simulate_period(
    w_start: pd.Series,
    period_returns: pd.DataFrame,
    cost_first_month: float,
) -> tuple[pd.Series, pd.Series]:
    """Simula l'evoluzione mese per mese dei pesi e dei rendimenti netti di
    portafoglio all'interno di un periodo tra due ribilanciamenti.

    I pesi "driftano" naturalmente con i rendimenti degli asset (nessun
    ribilanciamento infra-periodo, come da prassi tra due date di
    ribilanciamento). Il costo di transazione del ribilanciamento viene
    sottratto UNA SOLA VOLTA, dal rendimento del primo mese del periodo.

    Ritorna:
      - port_returns: rendimenti netti mensili di portafoglio nel periodo
      - w_end: pesi "drifted" all'ultimo mese del periodo (punto di partenza
               per il calcolo del turnover al ribilanciamento successivo)
    """
    w = w_start.copy()
    port_returns = pd.Series(index=period_returns.index, dtype=float)

    for j, (date, r) in enumerate(period_returns.iterrows()):
        gross_r = float((w * r).sum())
        net_r = gross_r - (cost_first_month if j == 0 else 0.0)
        port_returns.loc[date] = net_r

        # Drift dei pesi con i rendimenti lordi degli asset, poi rinormalizzo
        # a somma 1. Il costo e' trattato come un semplice "prelievo"
        # contabile sul rendimento riportato, non come cash drag sui pesi:
        # semplificazione standard per un progetto didattico.
        w = w * (1.0 + r)
        total = w.sum()
        if total > 0:
            w = w / total

    return port_returns, w


# ---------------------------------------------------------------------------
# 4. Motore walk-forward per UNA strategia
# ---------------------------------------------------------------------------
def run_backtest(
    returns: pd.DataFrame,
    riskfree: pd.Series,
    method: str,
    charge_initial_cost: bool = False,
    verbose: bool = False,
) -> dict:
    """Esegue il backtest walk-forward completo per UNA strategia.

    method in {"min_variance", "max_sharpe", "michaud_resampled"}.

    Ad ogni ribilanciamento in posizione i:
      - la finestra di stima usa solo returns.iloc[:i] (vedi estimation_window);
      - il risk-free per max-Sharpe e' l'ULTIMO valore disponibile PRIMA del
        ribilanciamento (riskfree.iloc[i-1]), mai un valore futuro;
      - i pesi ottimi vengono applicati ai rendimenti fuori campione del
        periodo successivo, con drift infra-periodo e costo di transazione
        sul turnover registrato.

    charge_initial_cost=False (default) -> il primo investimento (da cash a
    portafoglio) non viene conteggiato come costo di transazione, coerente
    con l'idea che non c'e' un portafoglio "precedente" da ribilanciare.

    Ritorna un dizionario con:
      - "returns"  : Serie dei rendimenti netti mensili di portafoglio
      - "weights"  : DataFrame dei pesi TARGET ad ogni data di ribilanciamento
      - "turnover" : Serie del turnover ad ogni ribilanciamento
      - "costs"    : Serie dei costi di transazione ad ogni ribilanciamento
    """
    n_obs = len(returns)
    rebal_idx = rebalance_indices(n_obs)
    dates = returns.index

    all_port_returns = []
    weights_history = {}
    turnover_history = {}
    cost_history = {}

    w_prior_drifted = None  # pesi "drifted" alla fine del periodo precedente

    for k, i in enumerate(rebal_idx):
        window = estimation_window(returns, i)
        rf_decision = float(riskfree.iloc[i - 1]) if i - 1 >= 0 else 0.0

        if method == "michaud_resampled":
            w_target = opt.michaud_resampled(window, method="max_sharpe", rf=rf_decision)
        else:
            mu, sigma = estim.estimate(window)
            w_target = opt.optimize(mu, sigma, method=method, rf=rf_decision)

        is_first = (k == 0)
        turn = turnover(w_target, w_prior_drifted)
        cost = transaction_cost(turn) if (not is_first or charge_initial_cost) else 0.0

        weights_history[dates[i]] = w_target
        turnover_history[dates[i]] = turn
        cost_history[dates[i]] = cost

        j_end = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else n_obs
        period_returns = returns.iloc[i:j_end]

        port_returns, w_prior_drifted = _simulate_period(w_target, period_returns, cost)
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
# 5. Tutte le strategie attive, sullo stesso periodo out-of-sample
# ---------------------------------------------------------------------------
def run_all_strategies(
    returns: pd.DataFrame,
    riskfree: pd.Series,
    strategies: list[str] | None = None,
    verbose: bool = False,
) -> dict[str, dict]:
    """Esegue run_backtest per ogni strategia in `strategies` (default:
    config.STRATEGIES), sulla STESSA serie di rendimenti/risk-free, cosi'
    da avere periodi out-of-sample perfettamente confrontabili.

    Ritorna {nome_strategia: risultato di run_backtest(...)}.
    """
    strategies = strategies or cfg.STRATEGIES
    results = {}
    for method in strategies:
        if verbose:
            print(f"\n--- Backtest: {method} ---")
        results[method] = run_backtest(returns, riskfree, method=method, verbose=verbose)
    return results


def combine_returns(results: dict[str, dict]) -> pd.DataFrame:
    """Affianca i rendimenti netti mensili di piu' strategie/benchmark in un
    unico DataFrame (colonna = nome strategia), pronto per metrics.py e
    plotting.py.

    `results` ha lo stesso formato del ritorno di run_all_strategies (o puo'
    essere assemblato a mano unendo strategie e benchmark, cfr. benchmarks.py).
    """
    return pd.DataFrame({name: res["returns"] for name, res in results.items()})


if __name__ == "__main__":
    # Piccolo self-test manuale: usa la cache di data_loader se presente.
    from data_loader import load_data

    data = load_data(use_cache=True)
    returns = data["returns"]
    riskfree = data["riskfree"]

    print(f"Periodo disponibile: {returns.index.min().date()} -> "
          f"{returns.index.max().date()} ({len(returns)} mesi)")
    print(f"Prima data di ribilanciamento: "
          f"{returns.index[_first_rebalance_idx(len(returns))].date()}")

    results = run_all_strategies(returns, riskfree, verbose=True)
    combined = combine_returns(results)

    print("\n=== Rendimenti netti mensili (testa) ===")
    print(combined.head())
    print("\n=== Turnover medio per strategia ===")
    for name, res in results.items():
        print(f"  {name}: {res['turnover'].mean():.3f}")
