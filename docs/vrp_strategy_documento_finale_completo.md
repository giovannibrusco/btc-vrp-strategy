# VRP Strategy — Documento Finale Completo

Questo documento integra e sostituisce i due precedenti (vrp_strategy_final.md e vrp_strategy_post_audit_final.md). Contiene l'intero percorso del progetto, tutti i risultati, e lo stato finale della strategia. È il documento unico di riferimento per la compilazione della tesi.

---

## PARTE I — IL PERCORSO DI RICERCA

### 1. Ipotesi 1: Gap RND-KDE come segnale di selezione (SCARTATA)

**L'idea**: Il gap pointwise tra la densità risk-neutral (RND, estratta dalle IV delle opzioni via Breeden-Litzenberger) e la densità fisica (KDE, rolling 365 giorni sui rendimenti BTC) identifica le opzioni più mispriced. Score = gap × vega.

**Pipeline costruita**: Signal → execution → hedge → exit → audit. T+1 timing corretto, no forward-looking bias. 6×5 sensitivity grid, tutti i cell profittevoli IS e OOS.

**Test critico**: Regressione del demeaned gap su realized PnL per trade.
- Perché demeaned: il gap grezzo è positivo in media (la IV quasi sempre supera la RV — è il VRP stesso). Testare il gap grezzo risponde alla domanda "esiste il VRP?" (trivialmente sì), non "il mio segnale discrimina tra momenti buoni e cattivi per venderlo?"
- Metodo: expanding mean per demeaning (strettamente causale, no parametri), clustered standard errors per data
- Risultato: β negativo e non significativo. Il gap non ha potere predittivo.

**Benchmark**: Top Vega (vendi l'opzione con vega più alto) batte la strategia complessa con meno trade e Sharpe migliore.

**Diagnosi**: Il confronto pointwise di due densità stimate con errore (RND e KDE) produce un segnale dominato dal rumore. La KDE flat (i.i.d., 365 giorni, Silverman bandwidth) è troppo imprecisa sulle code. Anche con una P-measure migliore (GARCH), il problema fondamentale del confronto pointwise rimarrebbe.

**Decisione**: Abbandonare il gap come segnale di selezione. La RND resta come infrastruttura tecnica e contesto di ricerca.

---

### 2. Ipotesi 2: Premium/DTE come strategia complementare (SCARTATA)

**L'idea**: Selezionare l'opzione con il più alto rapporto bid×spot/DTE — massimizzare l'efficienza del premium per unità di tempo.

**Risultati iniziali**: Sharpe 1.67 IS, 126 trade IS, 67 trade OOS. Combinata con Top Vega (zero overlap, correlazione PnL 0.27) formava un portafoglio diversificato con Sharpe 1.77 IS.

**L'audit ha rivelato tre problemi:**

1. **CRITICO — Bug bid=0 su opzioni illiquide**: Lo scoring favoriva opzioni deep ITM (delta mediano 0.99) perché hanno bid altissimi. Ma queste opzioni su Deribit sono completamente illiquide — nessuno le quota, il bid nei dati è 0. Il backtest vedeva bid=0 all'exit, scattava il profit target (100% del premium "catturato"), e registrava profitti inesistenti.
   - 33 trade su 126 (26%) avevano exit_price=0
   - Questi 33 trade generavano $351k — il 116.6% del totale ($301k)
   - I restanti 93 trade perdevano -$50k in aggregato
   - Tutta la profittabilità era un artefatto

2. **SERIO — Exit al bid invece che all'ask**: Quando ricompri una short option paghi l'ask, non il bid. Il backtest sottostimava sistematicamente il costo di uscita.

3. **MODERATO — Zero costi opzioni**: Le opzioni non avevano fee. Deribit applica 0.03% del sottostante per side (con cap al 12.5% del prezzo dell'opzione).

**Fix implementati**: Delta filter |delta| ≤ 0.50, exit con ask_price, costi opzioni Deribit. Applicati a backtest, live scorer, position manager, live trader.

**Risultati post-fix**: Sharpe IS = -0.24. Strategia morta.

**Nota sulla diagnosi iniziale errata**: La prima interpretazione era che Premium/DTE fosse una "scommessa direzionale" perché selezionava opzioni deep ITM. Questo era impreciso — se il delta hedge funziona, anche un'opzione deep ITM è delta-neutral. Il vero problema era il bid=0 nei dati. La diagnosi corretta è emersa investigando il breakdown del P&L dei mega-trade.

---

### 3. Ipotesi 3: Multi-position per aumentare il campione (SCARTATA)

**Il problema**: Top Vega produce ~35 trade IS in ~800 giorni di trading. Indagine: la strategia è in posizione il 92.8% del tempo (61 giorni flat su 851). Il driver è il holding period lungo (DTE entry mediano 66 giorni, holding medio 25.5 giorni) combinato con il vincolo di posizione singola. Non mancano segnali — manca capacità.

**Tentativo**: Fino a 3 posizioni simultanee con diversificazione per expiry bucket (short ≤21 DTE, medium 22-45 DTE, long >45 DTE).

**Risultati**:

| Metrica | Single IS | Multi IS | Single OOS | Multi OOS |
|---------|-----------|----------|------------|-----------|
| Trades | 35 | 147 | 10 | 60 |
| Sharpe | 1.74 | 0.63 | 1.53 | 1.76 |
| Return | 37% | 33% | 14% | 37% |
| MTM DD | -18.4% | -49.7% | -4.5% | -14.5% |

Il multi-position funziona in OOS (mercato calmo) ma è devastante in IS (bear market 2022). Il drawdown di -49.7% è inaccettabile. Le posizioni parallele sono tutte esposte allo stesso BTC — durante un crash perdono tutte insieme. L'hedge aggregato performa peggio (da +$10k a -$25k IS).

**Decisione**: Single-position. La bassa frequenza è una caratteristica della strategia.

---

### 4. Regime Detection — Cinque indicatori testati (NESSUNO PREDITTIVO)

**Motivazione**: La regime analysis mostra che la strategia è forte in low/mid vol e debole in high vol. Un filtro che riduce la size o non trada in high vol potrebbe migliorare il risk-adjusted return.

**Approccio**: Per ciascuno dei 35 trade IS, registrare il valore dell'indicatore all'entry e testare la correlazione con il PnL del trade.

**Indicatori testati e risultati**:

| Indicatore | Correlazione con PnL | Media Winners | Media Losers | Conclusione |
|------------|---------------------|---------------|-------------|-------------|
| DVOL (IV implicita BTC) | +0.202 | 76.4% | 74.6% | Non discrimina |
| Parkinson RV30 | +0.237 | 67.9% | 65.2% | Non discrimina |
| VRP Spread (DVOL - RV30) | -0.133 | 8.5pp | 9.4pp | Non discrimina (segno controintuitivo) |
| RND Skewness | -0.009 | -0.038 | -0.027 | Essenzialmente zero |
| RND Kurtosis | -0.086 | 2.753 | 2.573 | Essenzialmente zero |

**Osservazione chiave**: Trade con VRP spread alto (+12pp, +31pp) che perdono comunque. Il VRP all'entry non è il driver del PnL per trade. Il driver è probabilmente cosa succede *durante* il trade (jump improvvisi, esplosione di vol), non le condizioni all'ingresso.

**Razionale economico sulla scelta degli indicatori**:
- DVOL: livello della IV, misura quanto il mercato prezza la vol futura
- Parkinson RV30: livello della vol realizzata, cattura i range intraday (più informativo del close-to-close per BTC 24/7)
- VRP Spread: misura diretta del premio che si sta vendendo. Concettualmente il più rilevante — se è invertito (RV > IV) non c'è premio da catturare. Ma empiricamente non discrimina.
- RND Skewness: rischio di coda prezzato dal mercato. Se molto negativa, il mercato prezza crash risk.
- RND Kurtosis: pesantezza delle code della distribuzione implicita.

**Decisione**: Nessun filtro regime implementato. Con 35 trade e correlazioni tutte sotto 0.25, qualsiasi filtro sarebbe non testabile statisticamente.

**Letteratura considerata per future work**: GARCH(1,1) condizionale sulla P-measure, HAR-RV (Corsi 2009, componenti daily/weekly/monthly), Markov Regime Switching (Hamilton 1989), vol-of-vol, VRP come predittore dei propri rendimenti futuri (Bollerslev, Tauchen, Zhou 2009).

---

### 5. Stop Loss — Analisi esplorativa (CONTROINDICATO)

**Motivazione**: La strategia non ha stop loss. Una short vol senza stop è esposta a perdite potenzialmente grandi. Analisi necessaria.

**Metodo**: Per ciascuno dei 35 trade IS, calcolo del PnL unrealized MTM giorno per giorno, registrazione del worst unrealized PnL, confronto con il PnL finale.

**Finding principali**:

1. **I winner passano per drawdown significativi**: 15 su 23 winner hanno avuto worst unrealized < -$500. Caso estremo: worst -$17,386 (-15.5% del capitale) che poi ha chiuso a +$995.

2. **I loser passano molto più tempo in negativo**: media 59% dei giorni in negativo (15.1 giorni) vs 29% per i winner (6.2 giorni).

3. **7 trade erano in profitto (>$500) ma hanno chiuso in perdita** — il profit target non si è attivato in tempo.

**Test quantitativo sullo stop loss**:

| Stop Level | Trade colpiti | Winner persi | PnL winner perso | PnL loser "salvato" |
|------------|--------------|--------------|-----------------|-------------------|
| -1.0% | 15 | 8W + 7L | $23,380 | -$6,692 |
| -2.0% | 5 | 3W + 2L | $14,312 | -$2,928 |
| -3.0% | 3 | 2W + 1L | $8,619 | -$182 |
| -5.0% | 2 | 2W + 0L | $8,619 | $0 |

**A ogni livello di stop, il PnL perso dai winner stoppati supera il PnL risparmiato dai loser.** Sotto il -4% si stopperebbero solo winner.

**Spiegazione economica**: La strategia short vol ha bisogno di tempo per far lavorare il theta decay. I drawdown intraposizione sono il costo per catturare il premium. Il payoff è asimmetrico nel tempo: perdi subito quando la IV sale (mark-to-market negativo), recuperi lentamente col theta. Stoppare prematuramente taglia il recupero.

**Decisione**: Nessuno stop loss. Le uniche regole di uscita sono il profit target a 50% e l'exit temporale a DTE ≤ 5.

---

## PARTE II — LA STRATEGIA FINALE

### 6. Configurazione

- **Selezione**: Top Vega — ogni giorno, seleziona l'opzione con il vega più alto nell'universo filtrato (|delta| ≤ 0.50)
- **Posizioni**: Single-position (max 1 aperta alla volta)
- **Entry**: Vendi al bid, T+1 execution
- **Exit**: Ricompra all'ask. Trigger: profit target 50% del premium incassato OPPURE DTE ≤ 5
- **Delta hedge**: Futures trimestrali Deribit, rebalancing giornaliero
- **Costi**: Option fees 0.03% per side con cap Deribit, hedge fees 0.10% per side, spread bid-ask incorporato (entry al bid, exit all'ask)
- **Nessuno stop loss**: Analisi esplorativa dimostra che è controindicato
- **Nessun filtro regime**: Cinque indicatori testati, nessuno predittivo

### 7. Money Management — Capital-Based Vega Sizing

**Evoluzione del sizing**:
1. Fixed 1 contratto → problematico perché il rischio per trade oscilla con il vega dell'opzione
2. Fixed vega target = 50 → normalizza il rischio per trade ma non si adatta al capitale
3. Capital-based vega target → il vega target è funzione del capitale corrente

**Formula**:
```
worst_loss_per_vega = 54.93  # dal backtest IS (worst trade: -$2,746 su 50 vega)
adjusted_loss = worst_loss_per_vega × buffer_multiplier
vega_target = (current_equity × risk_per_trade) / adjusted_loss
```

Il vega target si ricalcola a ogni trade. Cresce con il capitale (compounding), si riduce dopo le perdite (protezione drawdown).

**Applicazione**: Post-hoc rescaling. I trade (date entry, date exit, strumenti selezionati) sono identici in tutte le configurazioni. Cambia solo la size in dollari. Lo Sharpe è invariante — è un rapporto.

**Tre configurazioni presentate**:

| Configurazione | Risk/Trade | Buffer | Vega Iniziale | Ret IS | DD IS | Ret OOS | DD OOS | Sharpe IS | Sharpe OOS |
|----------------|-----------|--------|---------------|--------|-------|---------|--------|-----------|------------|
| Conservativa | 1% | 2.0x | 9.1 | +6.9% | -3.9% | +2.6% | -0.9% | 1.75 | 1.52 |
| Moderata | 2% | 1.5x | 24.3 | +19.3% | -10.3% | +7.0% | -2.3% | 1.75 | 1.52 |
| Aggressiva | 3% | 1.0x | 54.6 | +47.8% | -22.6% | +16.1% | -5.1% | 1.75 | 1.52 |

**Nota**: La configurazione aggressiva (vega iniziale 54.6) è quasi identica al vega fisso 50 precedente. Il vantaggio del capital-based è che scala automaticamente.

**Perché non altri metodi di MM**:
- Half-Kelly: richiede stime di win rate e payoff ratio — con 35 trade le stime sono troppo rumorose per un sizing ottimale (vedi sanity check sotto)
- Vol targeting: ha senso per portafogli multi-posizione con correlazioni variabili, non per single-position
- Drawdown scaling: già incorporato implicitamente nel capital-based sizing (capitale scende → vega target scende)

### 7.1 Half-Kelly come sanity check

Il Half-Kelly è stato calcolato come verifica di coerenza, non come metodo di sizing.

**Input (dai 35 trade IS)**:
- Win rate: 65.7%
- Payoff ratio: 2.57 (media win 1.76% / media loss 0.69%)

**Calcolo**:
- Kelly fraction = 0.657 - (1 - 0.657) / 2.57 = 52.4%
- Half-Kelly = 26.2%
- Vega target equivalente = (100,000 × 0.262) / 54.93 = ~477

**Confronto con le configurazioni**:

| Configurazione | Risk/Trade | Vega | vs Half-Kelly |
|----------------|-----------|------|---------------|
| Conservativa | 1% | 9.1 | 0.02x HK |
| Moderata | 2% | 24.3 | 0.05x HK |
| Aggressiva | 3% | 54.6 | 0.11x HK |
| Half-Kelly | 26.2% | 476.9 | 1.00x HK |

**Interpretazione**: Tutte le configurazioni sono estremamente conservative rispetto al Half-Kelly (anche l'aggressiva è solo l'11% del HK). Questa conservatività è intenzionale:
1. 35 trade sono pochi per stimare il Kelly con precisione — l'incertezza sugli input è enorme
2. Il Kelly assume distribuzioni stazionarie e indipendenza tra trade
3. Il worst case IS (trade #10, -15.5% unrealized) mostra code pesanti che il Kelly non cattura
4. In pratica nessun fondo opera oltre il 10-15% del Kelly

Il sanity check conferma che le configurazioni non sono irragionevolmente conservative — sono prudenti in modo proporzionato all'incertezza.

### 7.2 Margini Deribit e soglia di rovina

**Formula Standard Margin (short option su Deribit)**:
```
IM = [max(0.15 - OTM/spot, 0.10) + mark_price] × n_contracts × spot
   + 4% × hedge_notional (quarterly future)
```

**Margini calcolati sui 35 trade IS**:

| Configurazione | IM medio | IM worst-case | Min capitale (1.5x buffer) |
|----------------|----------|---------------|---------------------------|
| Conservativa (vega 9.1) | $1,672 | $6,205 | $9,308 |
| Moderata (vega 24.3) | $4,465 | $16,570 | $24,855 |
| Aggressiva (vega 54.6) | $10,033 | $37,232 | $55,847 |

**Soglia di rovina**: $10,000. Sotto questo livello non è possibile aprire neanche un trade nella configurazione più conservativa con Standard Margin. Con il capital-based sizing questa soglia vale per tutte le configurazioni — il sizing si adatta automaticamente al capitale corrente.

**Nota sul delta all'entry**: Il worst-case margin include un trade con delta 0.75 all'entry. Il filtro |delta| ≤ 0.50 è applicato al signal date (T), ma il delta si muove tra segnale ed esecuzione (T+1). Il trade aveva delta -0.495 al signal e -0.755 all'entry per un movimento BTC overnight. Questo è realistico e non evitabile.

### 7.3 Risk of Ruin — Simulazione Monte Carlo

10.000 percorsi di 200 trade ciascuno, capital-based sizing, soglia di rovina $10,000, tre scenari.

**Dove P(Ruin) supera l'1%**:

| Scenario | P(Ruin) > 1% a | Vega corrispondente |
|----------|---------------|-------------------|
| Stime IS (WR=65.7%, PR=2.57) | Mai (< 0.1% anche a risk 15%) | — |
| Pessimistico (WR=55%, PR=2.0) | risk = 13% | vega ~237 |
| Molto pessimistico (WR=50%, PR=1.5) | risk = 8% | vega ~146 |

**Le tre configurazioni hanno P(Ruin) = 0.00% in tutti e tre gli scenari.** Nessun percorso su 10.000 ha mai raggiunto la rovina a risk 1-3%.

**Il vincolo reale: il drawdown, non la rovina.** Max drawdown atteso (mediana e P95):

| Configurazione | Stime IS | Pessimistico | Molto Pessimistico |
|----------------|----------|-------------|-------------------|
| Conservativa (1%) | -7% (P95: -11%) | -10% (P95: -16%) | -14% (P95: -23%) |
| Moderata (2%) | -14% (P95: -21%) | -20% (P95: -30%) | -26% (P95: -42%) |
| Aggressiva (3%) | -20% (P95: -30%) | -28% (P95: -42%) | -37% (P95: -56%) |

**Come leggere la tabella**: Se scegli la configurazione moderata e il mondo reale è peggiore delle stime IS (scenario pessimistico), il drawdown al 95° percentile è -30%. La scelta della configurazione dipende dalla tolleranza al drawdown, non dal rischio di rovina.

### 8. Profilo tipico dei trade

| Metrica | Valore |
|---------|--------|
| DTE medio entry | ~66 giorni |
| Moneyness | ~1.04 (near-ATM) |
| IV media | ~63% |
| Holding period medio | ~25 giorni |
| Vega per contratto | ~49-98 |
| Meccanismo di profitto | VRP selling (profitto da IV > RV) |

### 9. Risultati finali (post-audit, post-fix)

**Contesto di mercato**:

| | IS (2021-06 → 2023-09) | OOS (2023-10 → 2024-09) |
|---|---|---|
| BTC | $36,677 → $26,916 (-27%) | $27,050 → $64,543 (+139%) |
| Range | $15,797 — $68,368 | $26,777 — $73,285 |
| RV30 media | 57% | 50% |
| Carattere | Bear market + range (crash 2022 -55%) | Strong bull market |

**L'IS è il test più duro e informativo. L'OOS Sharpe più alto è probabilmente inflazionato dal bull market con vol in calo.**

**Metriche strategia (configurazione moderata: risk 2%, buffer 1.5)**:

| Metrica | IS | OOS |
|---------|-----|------|
| Trades | 35 | 10 |
| Sharpe | 1.75 | 1.52 |
| Return | +19.3% | +7.0% |
| MTM Max DD | -10.3% | -2.3% |
| Win Rate | ~58-60% | ~58-60% |

### 10. Validazione statistica

**t-test**: IS significativo al 1% (p=0.007). OOS significativo al 10% (p=0.09) — 10 trade, potenza limitata.

**Bootstrap Sharpe CI (10.000 ricampionamenti)**:
- IS: [0.72, 2.73], P(Sharpe > 1) = 93%
- OOS: [0.25, 3.11], P(Sharpe > 1) = 90%
- Nessun CI include zero

**Outlier removal**:
- Top-3 rimossi: Sharpe IS = 1.25, OOS = 0.35 — ancora positivo
- Top-5 rimossi: Sharpe IS = 0.72 (positivo), OOS negativo (5 trade rimanenti, non interpretabile)

**Regime breakdown**:
- Volatilità: Forte in low e mid vol IS, debole in high vol — pattern classico del vol selling
- Trend: Funziona in bull e bear IS (70% WR entrambi). OOS forte in bull, negativa in sideways
- Call/Put: Bilanciata, put leggermente più profittevole IS

**Cost stress**: Profittevole anche a 10x i costi di hedging attuali.

### 11. Validazione cross-asset — SPY

La stessa strategia (Top Vega, stessi parametri, zero ottimizzazione) è stata testata su opzioni SPY con dati EOD su 14 anni. L'obiettivo non è presentare una strategia su equity, ma validare che il meccanismo (cattura del VRP via selezione top vega) non è un artefatto del campione piccolo su BTC.

**Audit dei costi**: Il primo backtest SPY usava erroneamente i costi Deribit (0.03% del sottostante per opzioni, 0.075% per hedge). Dopo audit, i costi sono stati corretti ai costi equity realistici ($0.65/contratto per opzioni, $0.01/share per hedge). L'impatto è stato enorme: i costi per trade sono passati da $140 a $9, e lo Sharpe da -0.89 a +0.79.

**Audit dei dati**: 3 record su ~9.5 milioni avevano IV negativa nel dataset sorgente, producendo delta spurii. Impatto: $-11 su $9,071 di PnL totale — trascurabile. Nessun bug di tipo bid=0 (l'unico trade con exit_price=0 era una put deep OTM a $0.01, caso legittimo).

**Confronto cross-asset**:

| Metrica | BTC IS (3.3y) | BTC OOS (1y) | SPY (14y) |
|---------|--------------|-------------|-----------|
| Trades | 35 | 10 | 147 |
| Win Rate | 65.7% | 70.0% | 66.0% |
| Sharpe | 1.75 | 1.52 | 0.79 |
| Return | 37.0% | 14.3% | 9.1% |
| Max DD | -7.0% | -4.7% | -2.3% |
| Entry IV media | 69% | — | 16% |
| Costi/trade | $100 | — | $9 |

**Decomposizione option-only vs hedge**:

| Metrica | BTC IS | SPY |
|---------|--------|-----|
| Win rate option-only | 82.9% | 82.3% |
| Win rate totale | 65.7% | 66.0% |
| Option PnL/trade | +$858 | +$200 |
| Hedge PnL/trade | +$298 | -$129 |

**Interpretazione**:

1. **Il win rate option-only è identico (82%)** — il meccanismo Top Vega cattura il VRP su entrambe le asset class con la stessa efficacia.

2. **La differenza è nell'hedge**: su BTC l'hedge guadagna (+$298/trade) grazie alla volatilità alta che genera profitto dal rebalancing gamma. Su SPY l'hedge perde (-$129/trade) perché la vol è bassa e i costi di rebalancing superano il profitto da gamma.

3. **Verifica: l'hedge non ha bias direzionale.** Test in mercato piatto (rendimento sottostante < 2% durante il trade): su BTC l'hedge genera +$771/trade anche senza trend — conferma che il profitto viene dal gamma rebalancing, non dall'esposizione direzionale. Su SPY in mercato piatto l'hedge perde -$518/trade — la vol realizzata è troppo bassa per generare profitto da rebalancing.

4. **Attribuzione del profitto su BTC**: l'81% del PnL totale (+$858 su +$1,057) viene dalla vendita di opzioni (VRP selling). Il 19% (+$298 su +$1,057) viene dall'hedge (gamma rebalancing). La strategia è genuinamente VRP selling — l'hedge è un amplificatore del rendimento su asset ad alta volatilità, non la fonte principale.

5. **Lo Sharpe BTC è 2x più alto (1.75 vs 0.79)** — il VRP su BTC è strutturalmente più grande grazie alla IV 4x più alta (69% vs 16%). Il premio incassato è proporzionalmente più grande, ma i costi non scalano altrettanto.

6. **SPY con 147 trade su 14 anni compensa il campione piccolo di BTC** — il win rate identico (66%) su un campione 4x più grande e un periodo che include COVID 2020, bear market 2022, e tightening Fed, conferma che il meccanismo non è un artefatto.

**Questo risultato non è presentato come una strategia equity.** Sharpe 0.79 e return 9.1% su 14 anni non è attraente come strategia stand-alone. Il valore è come validazione cross-asset del meccanismo che su BTC produce Sharpe 1.75.

---

## PARTE III — LIMITAZIONI E FUTURE WORK

### 12. Limitazioni da dichiarare nella tesi

**Statistiche**:
1. Campione piccolo su BTC: 35 trade IS, 10 trade OOS. I CI sono larghi. La regime analysis OOS non è interpretabile.
2. OOS corto e favorevole: bull market con vol in calo inflaziona lo Sharpe OOS.
3. Nessun test formale di data snooping (Hansen SPA) sulla strategia finale.

**Di esecuzione**:
4. Delta hedge giornaliero: BTC ha gamma e jump risk intraday. L'hedge giornaliero lascia esposizione residua.
5. Fill idealizzati: entry al bid, exit all'ask è corretto nella direzione ma non cattura partial fills e slippage.
6. Tail hedge non implementato: il costo del tail hedge non è incluso. Per short vol è un costo reale.

**Strutturali**:
7. Single-position limita la capacity: una posizione alla volta su un singolo sottostante.
8. Nessun controllo margine: il backtest non verifica se il margine Deribit è sufficiente.
9. Premium/DTE non supporta dynamic sizing (vega ≈ 0 sulle opzioni selezionate) — ma la strategia è stata abbandonata.
10. Stop loss controindicato: la strategia richiede tolleranza per drawdown intraposizione significativi.

### 13. Sviluppi futuri

**Miglioramento del segnale**:
- GARCH(1,1) condizionale sulla P-measure per riabilitare il gap come segnale di timing
- HAR-RV (Corsi 2009) — decompone la RV in componenti daily/weekly/monthly, forte su crypto in letteratura
- Ensemble di forecast di volatilità (media di più modelli tende a performare meglio di ciascuno singolarmente)

**Miglioramento del regime detection (con più dati)**:
- Markov Regime Switching (Hamilton 1989) con probabilità filtrate
- Vol-of-vol come indicatore di instabilità
- VRP come predittore dei propri rendimenti futuri (Bollerslev, Tauchen, Zhou 2009)

**Miglioramento dell'esecuzione**:
- Tail hedge realistico calcolato dalla vol surface
- Hedging più frequente (intraday)
- Controllo margine Deribit nel backtest

**Estensioni**:
- Applicazione a ETH (secondo mercato opzioni più liquido su Deribit — dati storici non disponibili gratuitamente al momento dello studio)
- Strategia straddle/strangle senza delta hedge su equity — la decomposizione SPY mostra che la gamba opzione è profittevole (82% WR) ma l'hedge è un costo netto. Eliminare l'hedge con una struttura auto-hedgiata potrebbe rendere la strategia profittevole su asset a bassa vol
- Multi-asset portfolio con correlazione gestita
- Ottimizzazione dei pesi del portafoglio (con campione più ampio)

---

## PARTE IV — STRUTTURA SUGGERITA PER LA TESI

### Sezione 1: Introduzione
- Il VRP su BTC: definizione, perché esiste, perché crypto
- Research question: si può catturare sistematicamente il VRP su opzioni BTC?

### Sezione 2: Dati e infrastruttura
- Deribit: opzioni, futures trimestrali, struttura del mercato
- Volatility surface construction
- RND extraction via Breeden-Litzenberger
- Realized vol: Parkinson estimator (motivazione: BTC 24/7, cattura range intraday)
- Walk-forward backtest framework

### Sezione 3: Il percorso di ricerca — ipotesi testate e scartate
- 3.1: Gap RND-KDE (fallito — no potere predittivo)
- 3.2: Premium/DTE (artefatto — bug bid=0 su opzioni illiquide)
- 3.3: Multi-position (fallito — drawdown inaccettabile)
- 3.4: Regime detection (cinque indicatori, nessuno predittivo)
- 3.5: Stop loss (controindicato — taglia più winner che loser)
- **Questa sezione è il cuore della tesi per un intervistatore**

### Sezione 4: La strategia finale — Top Vega
- Meccanismo: VRP selling puro
- Configurazione completa
- Profilo dei trade

### Sezione 5: Money Management
- Evoluzione: fixed → vega target → capital-based
- Formula e calibrazione dal worst case IS
- Tre configurazioni (conservativa/moderata/aggressiva)
- Sharpe invariante — il trade-off è solo return vs drawdown
- Half-Kelly come sanity check (conferma conservatività ragionevole)
- Margini Deribit e soglia di rovina
- Risk of Ruin Monte Carlo: P(ruin) = 0% su tutte le configurazioni, il vincolo è il drawdown

### Sezione 6: Validazione statistica
- t-test (significativo al 1% IS)
- Bootstrap Sharpe CI (nessun CI include zero)
- Outlier removal (robusto fino a top-3)
- Regime breakdown (debole in high vol — coerente)
- Cost stress (robusto a 10x costi)
- **Validazione cross-asset SPY**: stessa strategia, zero ottimizzazione, 147 trade su 14 anni, win rate identico (66%), Sharpe 0.79. Conferma che il meccanismo non è un artefatto del campione BTC. Decomposizione: gamba opzione identica (82% WR), differenza interamente nell'hedge (+$298/trade BTC vs -$129/trade SPY)

### Sezione 7: Risk management
- Delta hedge via quarterly futures (motivazione: evitare funding rate perpetual)
- No stop loss (motivazione empirica e economica)
- MTM drawdown come metrica primaria di rischio
- Limitazioni dell'hedge giornaliero

### Sezione 8: Limitazioni e sviluppi futuri
- Tutti i punti dalle Sezioni 11-12 di questo documento

### Sezione 9: Conclusioni
- Il VRP su BTC è reale e catturabile sistematicamente
- La strategia più semplice (top vega) batte approcci più sofisticati
- Il percorso dal complesso al semplice è esso stesso un risultato
- L'audit ha trasformato risultati apparentemente ottimi in risultati credibili
- Cinque indicatori di timing testati, nessuno predittivo — il timing del VRP rimane un problema aperto
- Validazione cross-asset: lo stesso meccanismo funziona su SPY (win rate 66%, 147 trade, 14 anni), confermando che il VRP è strutturale e non specifico di BTC. L'alpha è proporzionale alla IV — BTC con IV 4x più alta offre opportunità superiori

---

## Appendice — Riepilogo decisioni e razionale

| Decisione | Razionale |
|-----------|-----------|
| Gap RND-KDE scartato | β negativo e non significativo su demeaned gap vs PnL |
| Premium/DTE scartata | 116% dei profitti era artefatto di bid=0 su opzioni illiquide |
| Multi-position scartato | DD IS -49.7% — correlazioni esplodono nei crash |
| Filtro regime non implementato | 5 indicatori testati, nessuno con correlazione > 0.25 |
| Stop loss non implementato | A ogni soglia taglia più PnL dai winner che dai loser |
| Top Vega come strategia finale | Semplice, economicamente motivata (massimizza esposizione VRP), IS Sharpe 1.75 attraverso bear market 2022 |
| Single-position | Multi fallisce in stress; bassa frequenza è una caratteristica |
| Capital-based vega sizing | Si adatta al capitale corrente; Sharpe invariante; three configs per diversi appetiti al rischio |
| Half-Kelly come sanity check, non sizing | 35 trade: stime input troppo rumorose per Kelly ottimale. HK conferma che le configurazioni sono conservative (2-11% del HK) ma proporzionate all'incertezza |
| Soglia rovina $10k | Margine minimo Deribit per 1 trade conservativo con Standard Margin. Calcolato sui trade IS con buffer 1.5x |
| P(Ruin) = 0% su tutte le configurazioni | Monte Carlo 10.000 percorsi × 200 trade. Anche nello scenario molto pessimistico (WR=50%, PR=1.5) le configurazioni non raggiungono mai la rovina |
| Drawdown come vincolo, non rovina | La scelta tra configurazioni dipende dalla tolleranza al drawdown (P95: -11% a -56% a seconda di config e scenario), non dal rischio di rovina |
| Exit al bid → ask | Costo realistico di chiusura (ricompri all'ask) |
| Delta filter ≤ 0.50 | Esclude deep ITM illiquide con bid=0 |
| Parkinson RV30 | Più efficiente del close-to-close per BTC 24/7 (cattura range intraday) |
| Buffer 1.5x sul worst loss | 35 trade IS: il worst case storico quasi certamente non è il vero worst case |
| Profit target 50% | Parametro del backtest, non ottimizzato sulla grid |
| Exit DTE ≤ 5 | Evita gamma explosion vicino a scadenza |
| Quarterly futures per hedge | Evita funding rate dei perpetual (costoso in bull market) |
| Validazione SPY come cross-asset test | Stessa strategia senza ottimizzazione. Win rate identico (66%), Sharpe 0.79 su 147 trade e 14 anni. Conferma il meccanismo su asset class diversa |
| SPY come validazione, non come strategia | Sharpe 0.79 e 9.1% su 14 anni non è attraente stand-alone. Il valore è dimostrare che il meccanismo funziona cross-asset e compensare il campione piccolo BTC |
| Audit costi SPY | Primo backtest SPY usava costi Deribit (20x gonfiati). Dopo correzione a costi equity realistici: Sharpe da -0.89 a +0.79 |
| Hedge profittevole su BTC, costoso su SPY | BTC: hedge +$298/trade (vol alta → gamma rebalancing profittevole). SPY: hedge -$129/trade (vol bassa → costi > profitto gamma). Stessa meccanica, segno opposto |
