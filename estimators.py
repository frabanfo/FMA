"""
estimators.py
=============
Stima di mu (rendimenti attesi) e Sigma (matrice di covarianza) su una
finestra di rendimenti mensili.

Responsabilita' di questo modulo (e solo queste):
  1. stimare mu con media storica sulla finestra;
  2. stimare Sigma campionaria o con shrinkage di Ledoit-Wolf;
  3. fornire le versioni annualizzate di mu e Sigma (solo per reporting/plot,
     MAI per l'ottimizzazione, che lavora in unita' mensili come i rendimenti).

Principio anti-look-ahead: OGNI funzione qui dentro riceve una finestra di
rendimenti gia' tagliata (returns_window) e usa SOLO i dati contenuti in
quella finestra. Il motore walk-forward (backtest.py) e' responsabile di
passare, ad ogni data di ribilanciamento, esclusivamente i mesi passati
(rolling o expanding, cfr. config.WINDOW_TYPE) -> nessuna statistica
calcolata sull'intero campione arriva mai qui dentro.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import config as cfg


# ---------------------------------------------------------------------------
# 1. Stima di mu (rendimenti attesi)
# ---------------------------------------------------------------------------
def historical_mu(returns_window: pd.DataFrame) -> pd.Series:
    """Media storica (aritmetica) dei rendimenti mensili sulla finestra.

    E' lo stimatore piu' semplice e anche il piu' rumoroso: e' proprio la sua
    instabilita' a motivare sia la min-variance (che non usa mu) sia il
    resampling di Michaud (che ne mediano l'incertezza via Monte Carlo).
    """
    mu = returns_window.mean()
    mu.name = "mu"
    return mu


def estimate_mu(returns_window: pd.DataFrame, method: str | None = None) -> pd.Series:
    """Dispatcher per lo stimatore di mu, secondo config.MU_ESTIMATOR.

    Attualmente supportato solo "historical" (coerente con config.py).
    La funzione resta comunque un dispatcher esplicito, cosi' in futuro si
    puo' aggiungere es. uno stimatore shrinkage (James-Stein) senza toccare
    il resto della pipeline.
    """
    method = method or cfg.MU_ESTIMATOR
    if method == "historical":
        return historical_mu(returns_window)
    raise ValueError(f"estimate_mu: metodo '{method}' non riconosciuto.")


# ---------------------------------------------------------------------------
# 2. Stima di Sigma (matrice di covarianza)
# ---------------------------------------------------------------------------
def sample_cov(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Matrice di covarianza campionaria (stimatore classico, non distorto).

    Con finestre non molto piu' lunghe del numero di asset (es. 60 mesi per
    16 asset) e' mal condizionata/quasi singolare: la usiamo come termine di
    paragone per mostrare l'effetto dell'estimation risk rispetto a
    Ledoit-Wolf.
    """
    sigma = returns_window.cov()
    return sigma


def ledoit_wolf_cov(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Matrice di covarianza con shrinkage di Ledoit-Wolf (sklearn).

    Lo shrinkage "restringe" la covarianza campionaria verso un target
    strutturato (in sklearn: un multiplo della matrice identita'), riducendo
    la varianza della stima a costo di un po' di bias. E' lo stimatore di
    base del progetto (config.COV_ESTIMATOR = "ledoit_wolf") perche' e' molto
    piu' stabile della covarianza campionaria quando T non e' >> N.

    L'intensita' di shrinkage (tra 0 e 1, stimata automaticamente da sklearn
    col criterio di Ledoit-Wolf) e' salvata come attributo .shrinkage_ sul
    DataFrame ritornato, utile per diagnostica/presentazione.
    """
    lw = LedoitWolf().fit(returns_window.values)
    sigma = pd.DataFrame(
        lw.covariance_,
        index=returns_window.columns,
        columns=returns_window.columns,
    )
    sigma.attrs["shrinkage"] = lw.shrinkage_
    return sigma


def estimate_sigma(returns_window: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
    """Dispatcher per lo stimatore di Sigma, secondo config.COV_ESTIMATOR."""
    method = method or cfg.COV_ESTIMATOR
    if method == "sample":
        return sample_cov(returns_window)
    if method == "ledoit_wolf":
        return ledoit_wolf_cov(returns_window)
    raise ValueError(f"estimate_sigma: metodo '{method}' non riconosciuto.")


# ---------------------------------------------------------------------------
# 3. Orchestrazione: mu e Sigma insieme, su una finestra
# ---------------------------------------------------------------------------
def estimate(
    returns_window: pd.DataFrame,
    mu_method: str | None = None,
    cov_method: str | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Stima congiunta di mu e Sigma su una finestra di rendimenti mensili.

    Pensata per essere chiamata dal motore walk-forward ad ogni data di
    ribilanciamento, passando la sola finestra di rendimenti passati
    (returns_window = returns.loc[t_start:t_end], con t_end < data corrente).

    Controllo minimo di sanita': la finestra deve contenere almeno tanti
    mesi quanti asset, altrimenti la covarianza campionaria e' certamente
    singolare (per Ledoit-Wolf il problema e' attenuato ma un controllo
    esplicito evita errori silenziosi a valle, nell'ottimizzatore).
    """
    n_obs, n_assets = returns_window.shape
    if n_obs < n_assets:
        raise ValueError(
            f"estimate: finestra troppo corta ({n_obs} mesi) rispetto al "
            f"numero di asset ({n_assets}); rischio di Sigma singolare."
        )

    mu = estimate_mu(returns_window, method=mu_method)
    sigma = estimate_sigma(returns_window, method=cov_method)
    return mu, sigma


# ---------------------------------------------------------------------------
# 4. Annualizzazione (solo per reporting/plot, mai per l'ottimizzazione)
# ---------------------------------------------------------------------------
def annualize_mu(mu_monthly: pd.Series, periods_per_year: int | None = None) -> pd.Series:
    """Annualizzazione lineare di mu: mu_annua = mu_mensile * 12.

    Approssimazione lineare (coerente con quella usata per il risk-free in
    data_loader.py): adeguata per rendimenti mensili piccoli, ed e' la
    convenzione standard nei corsi di asset allocation. Se i rendimenti sono
    log-rendimenti (config.USE_LOG_RETURNS = True), la somma su 12 mesi e'
    ESATTA (non approssimata), perche' i log-rendimenti sono additivi nel
    tempo: mu*12 e' quindi il log-rendimento annuo atteso.
    """
    ppy = periods_per_year or cfg.PERIODS_PER_YEAR
    return mu_monthly * ppy


def annualize_sigma(sigma_monthly: pd.DataFrame, periods_per_year: int | None = None) -> pd.DataFrame:
    """Annualizzazione della covarianza: Sigma_annua = Sigma_mensile * 12.

    Vale sotto l'ipotesi (standard, gia' usata implicitamente nel resto del
    progetto) di rendimenti mensili i.i.d. nel tempo: la covarianza scala
    linearmente con il numero di periodi, esattamente come la varianza scala
    con il tempo in un random walk.
    """
    ppy = periods_per_year or cfg.PERIODS_PER_YEAR
    return sigma_monthly * ppy


if __name__ == "__main__":
    # Piccolo self-test manuale: usa la cache di data_loader se presente.
    from data_loader import load_data

    data = load_data(use_cache=True)
    returns = data["returns"]

    # Ultima finestra di ESTIMATION_WINDOW mesi disponibile (come nel walk-forward).
    window = returns.iloc[-cfg.ESTIMATION_WINDOW:]
    mu, sigma = estimate(window)

    print("=== mu (mensile, ultima finestra) ===")
    print(mu.round(4))
    print("\n=== mu annualizzata ===")
    print(annualize_mu(mu).round(4))
    print(f"\n=== Sigma: shrinkage Ledoit-Wolf = {sigma.attrs.get('shrinkage'):.3f} ===")
    print(sigma.round(5))
