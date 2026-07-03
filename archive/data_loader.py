"""
data_loader.py
==============
Scarico, ispeziono, pulisco i dati e li trasformo in rendimenti mensili.

Responsabilita' di questo modulo (e solo queste):
  1. scaricare i prezzi (Adj Close) degli ETF da Yahoo Finance;
  2. scaricare il tasso risk-free da FRED;
  3. fare un'ispezione/pulizia documentata (NaN, storia insufficiente, spike);
  4. convertire i prezzi in rendimenti MENSILI (log o semplici);
  5. salvare tutto in cache locale per riproducibilita' e velocita'.

Principio anti-look-ahead a monte: qui NON facciamo nessuna normalizzazione
"globale" che usi statistiche dell'intero campione (medie, dev.std). Quelle
verranno calcolate dentro il motore walk-forward, finestra per finestra.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

import config as cfg

# yfinance e pandas_datareader sono dipendenze esterne (vedi requirements.txt).
try:
    import yfinance as yf
except ImportError as e:  # pragma: no cover
    raise ImportError("Manca yfinance. Installa con: pip install yfinance") from e


# ---------------------------------------------------------------------------
# 1. Download prezzi
# ---------------------------------------------------------------------------
def download_prices(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Scarica i prezzi giornalieri aggiustati (Adj Close) da Yahoo Finance.

    Uso `auto_adjust=True`: i prezzi includono dividendi e split, quindi sono
    gia' "total return" -> i rendimenti che ne ricaviamo sono total return,
    coerenti con quelli di un benchmark total return (corretto per il confronto).

    Ritorna un DataFrame: righe = giorni, colonne = ticker, valori = prezzo.
    """
    tickers = tickers or cfg.TICKERS
    start = start or cfg.START_DATE
    end = end if end is not None else cfg.END_DATE

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False,
    )
    # yfinance ritorna colonne MultiIndex (campo, ticker) per piu' titoli.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:  # caso di un solo ticker
        prices = raw[["Close"]]
        prices.columns = tickers

    # Riordino le colonne secondo l'ordine dell'universo in config.
    prices = prices[[t for t in tickers if t in prices.columns]]
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


# ---------------------------------------------------------------------------
# 2. Ispezione e pulizia (documentate)
# ---------------------------------------------------------------------------
def inspect_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Ritorna una tabella di diagnostica per ogni ticker.

    Serve all'ispezione preliminare richiesta dalla traccia: prima data
    disponibile, numero di osservazioni, percentuale di NaN, e il rendimento
    giornaliero massimo/minimo (per individuare spike/split residui anomali).
    """
    daily = prices.pct_change()
    report = pd.DataFrame({
        "descrizione": [cfg.DESCRIPTIONS.get(t, "") for t in prices.columns],
        "prima_data": [prices[t].first_valid_index() for t in prices.columns],
        "ultima_data": [prices[t].last_valid_index() for t in prices.columns],
        "n_oss": prices.notna().sum().values,
        "pct_nan_%": (prices.isna().mean() * 100).round(2).values,
        "rend_gg_max_%": (daily.max() * 100).round(1).values,
        "rend_gg_min_%": (daily.min() * 100).round(1).values,
    }, index=prices.columns)
    return report


def clean_prices(
    prices: pd.DataFrame,
    max_ffill_days: int = 5,
    drop_threshold: float = 0.98,
) -> pd.DataFrame:
    """Pulizia documentata dei prezzi.

    Passi (in ordine), tutti conservativi e privi di look-ahead:
      1. Scarto i ticker con copertura insufficiente sul periodo scelto
         (meno di `drop_threshold` di osservazioni rispetto al massimo).
      2. Tengo le sole righe in cui TUTTI i ticker superstiti sono quotati
         (allineamento del calendario su date comuni). Per dati mensili e
         un universo che parte tutto entro il 2007 questo taglia solo i primi
         giorni di rodaggio.
      3. Forward-fill LIMITATO (max `max_ffill_days` giorni) per coprire
         festivita'/giorni mancanti isolati, senza inventare lunghe serie.

    Ritorna i prezzi puliti e logga cosa e' stato fatto.
    """
    n0_cols, n0_rows = prices.shape[1], prices.shape[0]

    # (1) scarto ticker poco coperti
    coverage = prices.notna().mean()
    keep = coverage[coverage >= drop_threshold].index.tolist()
    dropped = [t for t in prices.columns if t not in keep]
    if dropped:
        warnings.warn(f"clean_prices: scartati per copertura < {drop_threshold:.0%}: {dropped}")
    prices = prices[keep]

    # (3) forward-fill limitato (applicato prima del dropna comune)
    prices = prices.ffill(limit=max_ffill_days)

    # (2) tengo solo le date in cui tutti i ticker superstiti hanno prezzo
    prices = prices.dropna(how="any")

    print(
        f"clean_prices: colonne {n0_cols}->{prices.shape[1]}, "
        f"righe {n0_rows}->{prices.shape[0]} "
        f"({prices.index.min().date()} -> {prices.index.max().date()})"
    )
    return prices


# ---------------------------------------------------------------------------
# 3. Da prezzi giornalieri a rendimenti mensili
# ---------------------------------------------------------------------------
def to_monthly_returns(prices: pd.DataFrame, log: bool | None = None) -> pd.DataFrame:
    """Campiona i prezzi a fine mese e calcola i rendimenti mensili.

    log=True  -> rendimenti logaritmici  r = ln(P_t / P_{t-1})
    log=False -> rendimenti semplici      r = P_t / P_{t-1} - 1

    Usiamo il mese come unita' (la traccia vieta il daily): a frequenza mensile
    l'ipotesi i.i.d. e' piu' ragionevole e l'annualizzazione e' piu' robusta.
    """
    log = cfg.USE_LOG_RETURNS if log is None else log
    monthly_prices = prices.resample(cfg.FREQ).last()
    if log:
        rets = np.log(monthly_prices / monthly_prices.shift(1))
    else:
        rets = monthly_prices.pct_change()
    return rets.dropna(how="any")


# ---------------------------------------------------------------------------
# 4. Risk-free da FRED
# ---------------------------------------------------------------------------
def load_riskfree(
    monthly_index: pd.DatetimeIndex,
    fred_code: str | None = None,
) -> pd.Series:
    """Scarica il risk-free (T-Bill 3m, % annua) da FRED e lo porta a frequenza
    mensile, allineato all'indice dei rendimenti.

    FRED fornisce un tasso ANNUO in percentuale (es. 5.0 = 5%). Lo converto in
    tasso mensile decimale dividendo per 100 e per 12 (approssimazione lineare,
    adeguata a tassi piccoli e coerente con l'annualizzazione lineare ×12 che
    usiamo per i rendimenti).

    Se FRED non e' raggiungibile, ritorna una serie di zeri (Sharpe con rf=0,
    come nei notebook del corso) e avvisa.
    """
    fred_code = fred_code or cfg.RISK_FREE_FRED_CODE
    try:
        from pandas_datareader import data as pdr
        start = monthly_index.min() - pd.Timedelta(days=40)
        rf_daily = pdr.DataReader(fred_code, "fred", start, monthly_index.max())
        rf_monthly_annual = rf_daily[fred_code].resample(cfg.FREQ).last() / 100.0
        rf_monthly = rf_monthly_annual / cfg.PERIODS_PER_YEAR
        rf = rf_monthly.reindex(monthly_index).ffill().fillna(0.0)
        rf.name = "rf"
        return rf
    except Exception as e:  # pragma: no cover
        warnings.warn(f"load_riskfree: FRED non disponibile ({e}); uso rf=0.")
        return pd.Series(0.0, index=monthly_index, name="rf")


# ---------------------------------------------------------------------------
# 5. Orchestrazione + cache
# ---------------------------------------------------------------------------
def load_data(use_cache: bool = True, refresh: bool = False) -> dict:
    """Funzione di alto livello: scarica/pulisce/converte e mette in cache.

    Ritorna un dizionario con:
      - "prices_daily" : prezzi giornalieri puliti
      - "returns"      : rendimenti mensili (il cuore di tutto)
      - "riskfree"     : risk-free mensile allineato ai rendimenti
      - "report"       : tabella di ispezione

    La cache (parquet) evita di riscaricare a ogni run e congela i dati ->
    risultati riproducibili. `refresh=True` forza il ridownload.
    """
    ret_cache = cfg.CACHE_DIR / "returns_monthly.parquet"
    px_cache = cfg.CACHE_DIR / "prices_daily.parquet"
    rf_cache = cfg.CACHE_DIR / "riskfree_monthly.parquet"

    if use_cache and not refresh and ret_cache.exists():
        returns = pd.read_parquet(ret_cache)
        prices = pd.read_parquet(px_cache)
        riskfree = pd.read_parquet(rf_cache)["rf"]
        print(f"load_data: cache caricata ({len(returns)} mesi, {returns.shape[1]} asset).")
        return {
            "prices_daily": prices,
            "returns": returns,
            "riskfree": riskfree,
            "report": inspect_prices(prices),
        }

    print("load_data: download da Yahoo Finance...")
    prices_raw = download_prices()
    report = inspect_prices(prices_raw)
    prices = clean_prices(prices_raw)
    returns = to_monthly_returns(prices)
    riskfree = load_riskfree(returns.index)

    # salvo in cache
    prices.to_parquet(px_cache)
    returns.to_parquet(ret_cache)
    riskfree.to_frame().to_parquet(rf_cache)
    print(f"load_data: salvato in cache -> {cfg.CACHE_DIR}")

    return {
        "prices_daily": prices,
        "returns": returns,
        "riskfree": riskfree,
        "report": report,
    }


if __name__ == "__main__":
    # Esecuzione diretta: scarica i dati e stampa un riepilogo.
    data = load_data(refresh=True)
    print("\n=== Report ispezione ===")
    print(data["report"])
    print("\n=== Rendimenti mensili (testa) ===")
    print(data["returns"].head())
    print(f"\nPeriodo: {data['returns'].index.min().date()} -> "
          f"{data['returns'].index.max().date()}  ({len(data['returns'])} mesi)")
