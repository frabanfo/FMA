# Dynamic Asset Allocation — Financial Markets Analytics 2026 (Variante B)

Backtest *walk-forward* out-of-sample di strategie di asset allocation dinamica
su un universo ETF multi-asset, con confronto tra Markowitz classico
(max-Sharpe, min-variance), resampling di Michaud e benchmark naïve (1/N, 60/40).

## Struttura

```
dynamic_asset_allocation/
├── config.py          # TUTTE le scelte: universo, periodo, parametri, seed
├── data_loader.py     # download (yfinance/FRED), pulizia, rendimenti mensili, cache
├── estimators.py      # μ (media storica), Σ (sample / Ledoit-Wolf)        
├── optimizers.py      # min-var, max-Sharpe (scipy), resampling Michaud    
├── backtest.py        # motore walk-forward (cuore del progetto)           
├── benchmarks.py      # 1/N, 60/40                                          [to do]
├── metrics.py         # equity, vol, max drawdown, Sharpe, Calmar, turnover [to do]
├── plotting.py        # grafici                                             [to do]
├── notebooks/         # 01_data, 02_optimizers_demo, 03_results (presentazione)
├── tests/             # sanity test, incl. verifica no-look-ahead
├── docs/              # guida teorica e spiegazione del progetto
├── data/cache/        # dati scaricati (parquet) — riproducibilità
├── results/           # figure e tabelle prodotte
└── requirements.txt
```

## Setup (Windows / PowerShell)

```powershell
# 1. Crea l'ambiente virtuale (Python 3.12)
py -3.12 -m venv .venv

# 2. Attivalo
.\.venv\Scripts\Activate.ps1

# 3. Installa le dipendenze
pip install -r requirements.txt
```

## Uso

```powershell
# Scarica e prepara i dati (crea la cache in data/cache/)
python data_loader.py
```

Stato attuale: **dati pronti e verificati** — 16 ETF, 222 mesi
(gen 2008 → giu 2026). I moduli di ottimizzazione e backtest sono in sviluppo
secondo il piano a milestone.

## Riproducibilità
- Seed unico in `config.py` (`SEED = 42`), usato dal resampling Monte Carlo.
- Dati congelati in cache parquet: stessi input → stessi output.
- Tutte le scelte soggettive centralizzate in `config.py`.
