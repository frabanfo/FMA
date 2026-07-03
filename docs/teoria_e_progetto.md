# Dynamic Asset Allocation — Guida teorica e spiegazione del progetto

*Financial Markets Analytics 2025/2026 — Variante B*

> **Nota (luglio 2026).** Il progetto è evoluto: l'implementazione di riferimento è ora il
> notebook unificato `AI_Concentration_Dynamic_Allocation.ipynb` (tema: la crisi
> AI/semiconduttori del 2026), non più i moduli `.py` (spostati in `archive/`). La teoria
> qui sotto (Markowitz, estimation risk, Michaud, backtest walk-forward) resta valida e
> utile per l'orale; i riferimenti "da dove viene nel codice" vanno letti come sezioni del
> notebook. Per Black–Litterman (ora strategia centrale, con tre view fondate sulla crisi)
> vedi la Sezione 11 del notebook e Idzorek (2005).

Questo documento è la "spina dorsale" concettuale del progetto: collega la
**teoria** (Markowitz, estimation risk, Michaud) al **codice** che scriviamo.
È pensato per studiare in parallelo allo sviluppo e per preparare l'orale.
Ogni sezione finisce con **"Da dove viene nel codice"** e **"Cosa potrebbero
chiederti"**.

---

## 0. L'idea in una frase

Costruiamo portafogli multi-asset, li **ribilanciamo periodicamente usando solo
l'informazione disponibile in quel momento** (backtest *walk-forward*,
out-of-sample), e confrontiamo: Markowitz classico (max-Sharpe e min-variance),
la sua versione robusta (resampling di Michaud) e benchmark naïve (60/40, 1/N).
Misuriamo per tutti: equity curve, volatilità, max drawdown, Sharpe.

La domanda di ricerca è: **la sofisticazione (Markowitz, poi Michaud) batte
davvero il portafoglio ingenuo, una volta tenuto conto dell'errore di stima e
dei costi?** Spesso la risposta è sorprendente — ed è proprio il punto.

---

## 1. Markowitz mean-variance (1952)

### Teoria
Un investitore sceglie i pesi `w` (quanto in ogni asset) bilanciando
**rendimento atteso** e **rischio** (varianza). Dati:
- `μ` = vettore dei rendimenti attesi (uno per asset);
- `Σ` = matrice di covarianza dei rendimenti.

Rendimento e varianza del portafoglio:
```
rendimento_p = wᵀ μ
varianza_p   = wᵀ Σ w
```

Tre "portafogli notevoli" che implementiamo:

1. **Minimum-variance con target return** (forma della traccia):
   ```
   min  wᵀ Σ w     s.t.   wᵀ μ = μ_target ,   wᵀ 1 = 1
   ```
   Variando `μ_target` si traccia la **frontiera efficiente**.

2. **Global Minimum-Variance (GMV)**: il punto più a sinistra della frontiera.
   ```
   min  wᵀ Σ w      s.t.   wᵀ 1 = 1
   ```
   *Non usa μ.* Per questo è la strategia più robusta all'estimation risk:
   l'unico input rumoroso (i rendimenti attesi) sparisce.

3. **Max-Sharpe (tangency)**: massimizza il rendimento per unità di rischio.
   ```
   max  (wᵀ μ − r_f) / √(wᵀ Σ w)
   ```
   È il portafoglio "ottimo" della teoria, ma anche **il più fragile**: dipende
   pesantemente da `μ`, che è notoriamente difficile da stimare.

### Vincoli che imponiamo
- **Long-only** (`w ≥ 0`): no short selling — realistico per un fondo retail.
- **Fully invested** (`wᵀ 1 = 1`): no leverage, no cash residuo.
- **Cap per asset** (`w ≤ 0.35`): impedisce concentrazioni estreme.

I vincoli riducono la frontiera teorica ma **stabilizzano** le soluzioni:
senza di essi, max-Sharpe produce pesi enormi e short aggressivi su asset con
μ stimato (per caso) alto.

### Da dove viene nel codice
`optimizers.py`: `min_variance()`, `max_sharpe()` risolti con
`scipy.optimize.minimize(method="SLSQP", ...)`, esattamente nello stile del
notebook del corso `Markowitz_etf.ipynb` (funzioni `portfolio_variance`,
`neg_sharpe`, vincoli come lista di dict, `bounds=[(0,1)]*n`).

### Cosa potrebbero chiederti
- *Perché annualizzi ×12 e non ×252?* Perché lavoriamo su rendimenti mensili:
  più vicini a i.i.d./normali, meno microstructure noise, scaling più accurato.
- *Differenza tra GMV e max-Sharpe?* GMV ignora μ (solo Σ); max-Sharpe usa
  entrambi ed è molto più sensibile all'errore di stima.
- *Cosa cambia il vincolo no-short?* Accorcia la frontiera e la rende interna
  a quella unconstrained, ma riduce drasticamente la sensibilità ai dati.

---

## 2. Estimation risk — perché Markowitz "impazzisce"

È il concetto centrale del progetto (Michaud 1989, *"The Markowitz Optimization
Enigma: Is Optimized Optimal?"*).

`μ` e `Σ` non sono noti: li **stimiamo** da un campione finito, quindi sono
**rumorosi**. L'ottimizzatore di Markowitz è un *"error maximizer"*: tende a
sovrappesare gli asset il cui rendimento atteso è stato **sovrastimato per caso**
e a sottopesare quelli sottostimati. Il risultato:
- pesi **estremi** e **instabili** (cambiano molto a ogni ribilanciamento → alto
  turnover → alti costi);
- performance out-of-sample spesso **peggiore** del semplice 1/N.

Due rimedi che usiamo:

1. **Shrinkage della covarianza (Ledoit-Wolf)**: `Σ_shrink = (1−δ)·Σ_sample +
   δ·F`, dove `F` è un target strutturato e `δ∈[0,1]` è scelto in forma chiusa
   per minimizzare l'errore atteso. Stabilizza `Σ` quando la finestra di stima
   non è molto più lunga del numero di asset (la `Σ` campionaria è mal
   condizionata / quasi singolare).
2. **Resampling di Michaud** (sezione 3): media su molte stime simulate.

### Da dove viene nel codice
`estimators.py`: `estimate_mu()` (media storica) ed `estimate_sigma()` con
opzione `"ledoit_wolf"` (da `sklearn.covariance.LedoitWolf`) vs `"sample"`.
Confronteremo i due per **mostrare empiricamente** l'effetto dell'estimation risk.

### Cosa potrebbero chiederti
- *Perché usi la media storica per μ se è rumorosa?* Apposta: è il problema che
  vogliamo evidenziare. GMV e Michaud sono le risposte a questa fragilità.
- *Cos'è δ in Ledoit-Wolf?* L'intensità di shrinkage, stimata dai dati (non
  scelta a mano): bilancia varianza (Σ campionaria) e bias (target F).

---

## 3. Resampling di Michaud (Michaud & Michaud 2008)

### L'idea
Invece di ottimizzare una sola volta sui `μ, Σ` stimati (e amplificarne il
rumore), **simuliamo molti scenari coerenti con quelle stime**, ottimizziamo in
ciascuno, e **mediamo i pesi**. La media smussa l'errore di stima.

### Procedura (quella che implementiamo)
Per `m = 1 … M` (es. M = 500):
1. simula una storia di rendimenti di lunghezza `T` (= finestra di stima)
   estraendo da una normale multivariata `N(μ̂, Σ̂)`;
2. ri-stima `μ_m, Σ_m` dai dati simulati;
3. risolvi l'ottimizzazione (es. max-Sharpe long-only) → pesi `w_m`;
4. accumula `w_m`.

Pesi finali: `w* = (1/M) Σ_m w_m` (poi rinormalizzati a somma 1).

Risultato: pesi **più diversificati, più stabili nel tempo, meno turnover**. La
"frontiera resampled" è interna a quella classica ma molto più robusta
out-of-sample.

### Da dove viene nel codice
`optimizers.py`: `michaud_resampled()` usa `np.random.multivariate_normal`
(seed da `config.SEED`) e riusa la stessa routine `max_sharpe()` dentro il loop.
Riferimento di stile: il notebook del corso `Frontier_Robust.ipynb`.

### Cosa potrebbero chiederti
- *Perché simuli da N(μ̂, Σ̂)?* È il modello generativo coerente con le nostre
  stime: rappresenta "altre possibili storie" compatibili con i dati.
- *Quante simulazioni M?* 500 come base: trade-off tra liscezza della media e
  costo computazionale (faremo un test di sensibilità).
- *Perché Michaud riduce il turnover?* Mediando, i pesi reagiscono meno agli
  shock di stima campione-per-campione → ribilanciamenti più dolci.

---

## 4. Black-Litterman (opzionale, Idzorek 2005)

Estensione *bonus* (la facciamo solo se siamo in anticipo). Parte dai
**rendimenti di equilibrio impliciti** `π` (reverse optimization dai pesi di
mercato) e li **combina con le view soggettive** dell'investitore:
```
E[R] = [ (τΣ)⁻¹ + Pᵀ Ω⁻¹ P ]⁻¹ · [ (τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q ]
```
- `P, Q` = matrice e vettore che codificano le view ("equity EM batterà govt
  del 2%");
- `Ω` = incertezza sulle view (più alta → view meno influente);
- `τ` = scalare che pesa la fiducia nell'equilibrio.

Se la includiamo, dovremo **dichiarare esplicitamente** come generiamo le view e
come calibriamo `Ω` e `τ` (lo richiede la traccia). Riferimento di stile: il
notebook del corso `Black_Litterman.ipynb`.

---

## 5. Metodologia walk-forward out-of-sample (il cuore del voto)

Questo è ciò che la traccia premia di più: **zero look-ahead bias**.

### Il ciclo
A ogni data di ribilanciamento `t`:
1. **stima**: usa solo i rendimenti fino a `t` (finestra rolling di 60 mesi).
   L'ottimizzatore non vede MAI dati futuri;
2. **decisione**: calcola i pesi `w_t`;
3. **incasso**: il portafoglio guadagna `w_t · r_{t+1}` (rendimento del periodo
   *successivo*). I pesi decisi a fine `t` fruttano nel mese `t+1`.

Separazione netta: *decidere* (solo passato) e *incassare* (solo futuro) sono
due fasi distinte nel codice, mai mescolate.

### Drift e costi di transazione
Tra un ribilanciamento e l'altro i pesi **driftano** con i rendimenti.
Al ribilanciamento successivo:
```
turnover_t = Σ_i | w_target,i − w_drifted,i |
costo_t    = c · turnover_t        (c = 10 bps base)
```
Il costo viene sottratto dal rendimento. Registriamo il turnover come metrica:
ci aspettiamo Markowitz classico con turnover alto, Michaud più basso.

### Salvaguardie anti-look-ahead (verificate da test)
- gli estimatori ricevono solo lo slice `R.loc[:t]`;
- Ledoit-Wolf è rifittato su ogni finestra (mai sull'intero campione);
- nessuna normalizzazione globale che usi statistiche full-sample;
- un test automatico altera i dati dopo `t` e verifica che `w_t` non cambi.

### Da dove viene nel codice
`backtest.py`: la funzione `walk_forward()`. È il modulo che **entrambi** i
membri del gruppo devono saper spiegare.

### Cosa potrebbero chiederti
- *Dove sarebbe il look-ahead se sbagliassi?* Esempi: stimare μ/Σ sull'intero
  campione; applicare `w_t` al rendimento di `t` invece di `t+1`; usare
  Ledoit-Wolf fittato una volta sola.
- *Rolling vs expanding window?* Rolling = adattiva (dimentica il passato
  remoto); expanding = usa tutto lo storico. Base rolling, test su expanding.

---

## 6. Metriche di performance (per ogni strategia E per i benchmark)

| Metrica | Formula (su rendimenti mensili) | Cosa dice |
|---|---|---|
| Equity curve | `cumprod(1 + r_p)` | crescita di 1$ investito |
| Rendimento annualizzato (CAGR) | da equity curve | rendimento medio composto |
| Volatilità | `std(r_p) · √12` | rischio |
| Sharpe | `mean(r_p − r_f)/std(r_p) · √12` | rendimento per unità di rischio |
| Max drawdown | min di `(equity/cummax − 1)` | peggior perdita picco-valle |
| Calmar | CAGR / |max drawdown| | rendimento per unità di drawdown |
| Turnover medio | media di `turnover_t` | costo/stabilità della strategia |

Tutte calcolate **anche per i benchmark**: solo così possiamo dire se la
strategia ha "creato valore" (richiesta esplicita della traccia, punto 7).

### Da dove viene nel codice
`metrics.py`: una funzione per metrica + `summary_table()` che le mette in una
tabella unica strategia-per-strategia.

---

## 7. Benchmark naïve (obbligatori)

- **1/N (equal-weight)**: stesso peso a tutti gli asset, ribilanciato alla
  stessa frequenza. È sorprendentemente difficile da battere (DeMiguel et al.
  2009) proprio perché immune all'estimation risk.
- **60/40**: 60% equity / 40% bond (equipesati dentro ogni sleeve). Il classico
  portafoglio bilanciato.

### Da dove viene nel codice
`benchmarks.py`: `equal_weight()`, `sixty_forty()`.

---

## 8. Universo e dati — giustificazione (traccia punto 4)

- **16 ETF USD multi-asset**: equity (US total, dev ex-US, EM, US small), govt
  (scaletta 1-3y/3-7y/7-10y/20y+), credito (IG, HY, EM bond), real assets (oro,
  commodity, REIT), inflation-linked (TIPS), cash (T-bill). Copre tutte le asset
  class richieste e oltre.
- **Perché ETF e non il dataset azionario fornito?** La DAA è per natura un
  problema *multi-asset*: con sole azioni la diversificazione tra classi
  sparisce. Il prof incoraggia esplicitamente un universo ETF proprio.
- **Periodo**: gen 2008 → oggi (~18 anni mensili), oltre i 15 richiesti. Include
  la crisi 2008 e il Covid 2020 come stress test del drawdown.
- **Fonte**: Yahoo Finance via `yfinance`, `Adj Close` (incorpora dividendi e
  split → total return). Risk-free: T-Bill 3m da FRED.
- **Prospettiva USD**: nessuna conversione valutaria → niente rischio cambio da
  modellare (scelta dichiarata e semplificante).
- **Pulizia documentata**: scarto ticker poco coperti, allineamento calendario
  su date comuni, forward-fill limitato (≤5 giorni). Vedi `data_loader.py`.

---

## 9. Narrativa per l'orale (filo conduttore)

Presenta i risultati in ordine di sofisticazione crescente:
```
1/N  →  60/40  →  GMV (min-var)  →  Markowitz max-Sharpe  →  Michaud
```
La storia da raccontare:
1. il naïve 1/N è un avversario tosto (estimation-risk-free);
2. Markowitz max-Sharpe "in teoria" è ottimo ma out-of-sample soffre l'errore di
   stima (pesi instabili, turnover alto, magari Sharpe peggiore);
3. GMV e soprattutto Michaud **recuperano** stabilizzando le stime;
4. i costi di transazione e la scelta della covarianza (sample vs Ledoit-Wolf)
   spostano le conclusioni → mostriamo le analisi di sensibilità.

Messaggio finale: **la robustezza alle stime conta più della "ottimalità"
teorica**. È esattamente la lezione di Michaud (1989).

---

## Riferimenti
- Markowitz H. (1952), *Portfolio Selection*, Journal of Finance.
- Michaud R. (1989), *The Markowitz Optimization Enigma: Is Optimized Optimal?*
- Michaud R. & Michaud R. (2008), *Estimation Error and Portfolio Optimization:
  A Resampling Solution*.
- Idzorek T. (2005), *A Step-by-step Guide to the Black-Litterman Model*.
- Ledoit O. & Wolf M. (2004), *Honey, I Shrunk the Sample Covariance Matrix*.
- DeMiguel, Garlappi, Uppal (2009), *Optimal Versus Naive Diversification*.
