"""
config.py
=========
Configurazione centrale del progetto di Dynamic Asset Allocation.

Qui vivono TUTTE le scelte soggettive del progetto (universo, periodo,
parametri di stima/ottimizzazione, costi). Tenerle in un unico file:
  - rende il progetto riproducibile (un solo posto da guardare/cambiare);
  - separa le DECISIONI dai MECCANISMI (il codice degli altri moduli non
    contiene "numeri magici": legge tutto da qui).

Nessuna logica di calcolo in questo file: solo costanti e piccole utility.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Riproducibilita'
# ---------------------------------------------------------------------------
# Un unico seed per tutto il progetto. Serve soprattutto al resampling di
# Michaud (Monte Carlo): senza seed i pesi cambierebbero a ogni esecuzione.
SEED = 42

# ---------------------------------------------------------------------------
# 1. Percorsi del progetto
# ---------------------------------------------------------------------------
# Tutti i path sono relativi a questo file -> il progetto e' portabile
# (funziona indipendentemente dalla cartella da cui lo lanci).
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"        # prezzi/rendimenti scaricati (cache locale)
RESULTS_DIR = ROOT_DIR / "results"    # figure, tabelle, pesi salvati
for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Universo investibile (ETF multi-asset, USD)
# ---------------------------------------------------------------------------
# 16 ETF quotati negli USA, denominati in USD, che coprono le principali
# asset class. Sono stati scelti perche':
#   - sono liquidi e rappresentativi della loro asset class;
#   - hanno storia sufficiente: TUTTI esistono entro fine 2007, quindi
#     possiamo partire da gennaio 2008 e avere >15 anni di dati mensili
#     (requisito della traccia: campione >= 15 anni).
#
# La prospettiva e' quella di un investitore in USD (Adj Close gia' in dollari):
# nessuna conversione valutaria, nessun rischio cambio da modellare.
#
# Struttura: ticker -> (descrizione, asset_class).
UNIVERSE: dict[str, tuple[str, str]] = {
    # --- Equity (azionario) ---
    "VTI": ("US Total Market Equity",        "equity"),
    "VEA": ("Developed ex-US Equity",        "equity"),
    "VWO": ("Emerging Markets Equity",       "equity"),
    "IWM": ("US Small Cap Equity",           "equity"),
    # --- Government bonds (titoli di stato, scaletta di duration) ---
    "SHY": ("US Treasury 1-3y",              "govt"),
    "IEI": ("US Treasury 3-7y",              "govt"),
    "IEF": ("US Treasury 7-10y",             "govt"),
    "TLT": ("US Treasury 20y+",              "govt"),
    # --- Credito (corporate / emergenti) ---
    "LQD": ("US Investment Grade Corp",      "credit"),
    "HYG": ("US High Yield Corp",            "credit"),
    "EMB": ("Emerging Markets Bond (USD)",   "credit"),
    # --- Real assets (oro, materie prime, immobiliare) ---
    "GLD": ("Gold",                          "real_asset"),
    "DBC": ("Broad Commodities",             "real_asset"),
    "VNQ": ("US Real Estate (REIT)",         "real_asset"),
    # --- Inflation-linked ---
    "TIP": ("US TIPS (inflation-linked)",    "inflation"),
    # --- Cash proxy ---
    "BIL": ("US T-Bill 1-3m (cash)",         "cash"),
}

# Liste/mappe derivate, comode da usare negli altri moduli.
TICKERS: list[str] = list(UNIVERSE.keys())
DESCRIPTIONS: dict[str, str] = {t: UNIVERSE[t][0] for t in TICKERS}
ASSET_CLASS: dict[str, str] = {t: UNIVERSE[t][1] for t in TICKERS}

# Raggruppamento ticker per asset class (utile per vincoli di budget e plot).
def tickers_by_class() -> dict[str, list[str]]:
    """Ritorna {asset_class: [ticker, ...]} a partire da UNIVERSE."""
    out: dict[str, list[str]] = {}
    for t, (_desc, cls) in UNIVERSE.items():
        out.setdefault(cls, []).append(t)
    return out

# ---------------------------------------------------------------------------
# 3. Periodo campionario e frequenza
# ---------------------------------------------------------------------------
# Partiamo a fine 2007 cosi' il primo rendimento mensile e' quello di gen-2008
# (tutti gli ETF sono gia' quotati). END = None -> "fino a oggi".
START_DATE = "2007-12-01"
END_DATE = None

# Lavoriamo a frequenza MENSILE (la traccia vieta il daily). Motivo teorico:
# i rendimenti mensili sono piu' vicini a i.i.d./normali e l'annualizzazione
# (x12 medie, x12 covarianza, sqrt(12) volatilita') e' molto piu' accurata
# che dal daily (dove il microstructure noise gonfia le stime).
FREQ = "ME"                 # "ME" = month-end (convenzione pandas, come nei notebook del prof)
PERIODS_PER_YEAR = 12       # fattore di annualizzazione per dati mensili
USE_LOG_RETURNS = True      # log-returns: comodi per aggregazione temporale

# ---------------------------------------------------------------------------
# 4. Parametri del backtest walk-forward
# ---------------------------------------------------------------------------
# Finestra di stima ROLLING di mu e Sigma, in MESI.
#   60 = 5 anni. Compromesso tra stabilita' (finestra lunga) e reattivita'
#   (finestra corta). Testeremo anche 36 come analisi di sensibilita'.
ESTIMATION_WINDOW = 60

# Ribilanciamento ogni quanti mesi. 3 = trimestrale (base).
#   Trimestrale = meno turnover/costi del mensile, ma comunque reattivo.
#   Testeremo anche 1 (mensile) come sensibilita'.
REBALANCE_EVERY = 3

# Tipo di finestra: "rolling" (lunghezza fissa) o "expanding" (ancorata).
WINDOW_TYPE = "rolling"

# ---------------------------------------------------------------------------
# 5. Vincoli di ottimizzazione
# ---------------------------------------------------------------------------
LONG_ONLY = True            # pesi >= 0 (no short selling)
FULLY_INVESTED = True       # somma pesi = 1 (no leverage, no cash residuo)
WEIGHT_CAP = 0.35           # cap massimo per singolo asset (None = nessun cap)
# Vincoli di budget per asset class (es. {"equity": (0.0, 0.7)}). None = nessuno.
CLASS_BUDGET: dict[str, tuple[float, float]] | None = None

# ---------------------------------------------------------------------------
# 6. Stimatori
# ---------------------------------------------------------------------------
# Stimatore della covarianza: "ledoit_wolf" (shrinkage, base) o "sample".
#   La matrice campionaria e' mal condizionata quando la finestra non e'
#   molto piu' lunga del numero di asset; lo shrinkage di Ledoit-Wolf la
#   stabilizza. Confronteremo i due per mostrare l'effetto dell'estimation risk.
COV_ESTIMATOR = "ledoit_wolf"

# Stimatore dei rendimenti attesi: media storica sulla finestra.
#   E' volutamente "rumoroso": e' proprio il problema che motivano Michaud
#   (resampling) e la min-variance (che non usa mu).
MU_ESTIMATOR = "historical"

# ---------------------------------------------------------------------------
# 7. Resampling di Michaud (Monte Carlo)
# ---------------------------------------------------------------------------
N_RESAMPLES = 500           # numero di path Monte Carlo (M). Piu' alto = piu' liscio, piu' lento.
# Lunghezza di ciascuna simulazione (in mesi). Convenzione Michaud: pari alla
# finestra di stima, cosi' l'incertezza simulata replica quella reale.
RESAMPLE_LENGTH = ESTIMATION_WINDOW

# ---------------------------------------------------------------------------
# 8. Costi di transazione e risk-free
# ---------------------------------------------------------------------------
# Costo proporzionale al turnover (L1) ad ogni ribilanciamento, in punti base.
#   10 bps = 0.0010. Testeremo 0 e 25 bps come sensibilita'.
TRANSACTION_COST_BPS = 10.0

# Serie risk-free da FRED per lo Sharpe (T-Bill 3 mesi, % annua).
#   DGS3MO e' il rendimento del Treasury a 3 mesi (constant maturity).
RISK_FREE_FRED_CODE = "DGS3MO"

# ---------------------------------------------------------------------------
# 9. Strategie e benchmark da eseguire
# ---------------------------------------------------------------------------
# Strategie "attive" che il motore ottimizzera' ad ogni ribilanciamento.
STRATEGIES = ["max_sharpe", "min_variance", "michaud_resampled"]

# Benchmark naive (richiesti dalla traccia), ribilanciati alla stessa frequenza.
BENCHMARKS = ["equal_weight", "sixty_forty"]

# Definizione del 60/40: quote per asset class (equity 60%, bond 40%).
# All'interno di ogni sleeve i pesi sono equipesati tra gli ETF di quella classe.
SIXTY_FORTY_EQUITY = ["VTI", "VEA", "VWO", "IWM"]
SIXTY_FORTY_BOND = ["SHY", "IEI", "IEF", "TLT"]
SIXTY_FORTY_SPLIT = (0.60, 0.40)
