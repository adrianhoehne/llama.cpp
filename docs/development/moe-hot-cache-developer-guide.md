# MoE Hot-Cache: Entwicklerdokumentation

Stand: 2026-05-14  
Branch: `cached-experts-v2`  
Letzter betrachteter Commit: `e3ace0d3b Separate moe hot cache feature pt5.`

Diese Dokumentation beschreibt die experimentelle MoE-Hot-Cache-Parallelisierung in diesem Fork: was geaendert wurde, warum es geaendert wurde, wie der Code zur Laufzeit arbeitet, welche Schalter relevant sind und wie die bisherigen Performance-Ergebnisse einzuordnen sind.

Der Fokus liegt auf Qwen3.5/Qwen3.6 MoE, insbesondere auf dem Ziel, Decode/TG durch eine Aufteilung der Expertenarbeit auf GPU und CPU schneller zu machen.

## Kurzfassung

Die Aenderung fuehrt einen optionalen Hot-Cache fuer MoE-Experten ein.

Ohne Hot-Cache laeuft der Server wie vorher. Erst wenn ein Hot-Cache-Budget gesetzt wird, wird der neue Pfad aktiv:

```text
--moe-hot-cache-max-mib > 0
```

Dann muss zusaetzlich eine Hot-Cache-JSON-Datei gesetzt sein:

```text
--moe-hot-cache <datei.json>
```

Der neue Pfad:

1. analysiert eine Perf-JSON mit Layer-/Expertennutzung,
2. waehlt pro Layer eine Menge "hot" Experten innerhalb eines Speicherbudgets,
3. kopiert diese hot Experten in einen eigenen Cache,
4. baut fuer aktive Layer einen speziellen Qwen-MoE-Graph,
5. trennt zur Laufzeit hot und cold Expertenarbeit,
6. fuehrt hot Arbeit auf der GPU und cold Arbeit auf der CPU parallel aus,
7. merged die Ergebnisse danach wieder in den normalen Tensorfluss.

Die wichtigsten Ziele waren:

- GPU-Auslastung im Decode besser nutzen,
- cold Expertenarbeit nicht laenger komplett seriell hinter der GPU ausfuehren,
- Overhead im Decode reduzieren,
- Perf-Messung auf Optimierungsdaten statt Visualisierungsdaten fokussieren,
- den experimentellen Code moeglichst getrennt vom upstream-nahen Kern halten.

## Ergebnisstand

Die Zahlen sind keine formalen Benchmarks, sondern Entwicklungslaeufe aus diesem Branch. Sie sind trotzdem hilfreich, weil sie die Richtung der Optimierung zeigen.

| Zustand | Beobachtung |
| --- | --- |
| Ohne Hot/Cold-Parallelisierung, einfacher "Hallo"-Test | ca. 19.7 tk/s |
| Erster paralleler PoC mit unguenstiger Hot-Liste | ca. 13.96 tk/s bei einfachem Prompt |
| Fruehe Programmiertasks | ca. 16 bis 17 tk/s |
| Nach Scheduler-, Routing- und Merge-Optimierungen | ueber 20 tk/s |
| Spaetere Performance-Laeufe `performance40` bis `performance52` | ca. 23.7 bis 25.5 tk/s |
| `performance53` | knapp unter 26 tk/s |
| `performance54` | ca. 27.1 tk/s |
| `performance55` | ca. 27.76 tk/s |
| `performance56` | ca. 28.09 tk/s |

Der groesste qualitative Fortschritt war nicht ein einzelner Patch, sondern die Kombination aus:

- weniger Decode-Routing-Overhead,
- weniger Merge-Overhead,
- weniger unnoetigen Branch-Starts,
- besserem Cold-Pfad,
- Hot/Cold-Regionen im Scheduler,
- reduzierter Perf-Messlast,
- sauberem Gating, damit der normale Pfad unberuehrt bleibt.

## Wichtige Begriffe

`hot experts`  
Experten, die laut Perf-Daten haeufig genug oder teuer genug sind, um in den GPU-Hot-Cache zu kommen.

`cold experts`  
Experten, die nicht in den Hot-Cache passen oder nicht ausgewaehlt wurden. Sie bleiben im normalen Modellpfad und werden im Parallelmodus bevorzugt auf der CPU verarbeitet.

`hot slot ratio`  
Anteil der MoE-Slots, die durch hot Experten bedient werden. Bei den bisherigen Daten lag die praktische Erwartung bei etwa 68 bis 70 Prozent. Deshalb muss der Overhead niedrig sein: die GPU kann nicht 100 Prozent der Expertenarbeit treffen.

`slot`  
Ein konkreter Token-Expert-Auftrag. Bei `n_expert_used = 8` entstehen pro Token bis zu acht Experten-Slots. Wird `n_expert_used` auf 16 erhoeht, verdoppelt sich diese Slot-Arbeit grob, sofern das Modell und die Architektur das zulassen.

`decode`  
Der Ein-Token-Pfad waehrend der Token-Generierung. Genau hier ist Overhead besonders kritisch.

`prefill`  
Der Prompt-Verarbeitungspfad mit vielen Tokens. Er hat andere Kostenverhaeltnisse als Decode.

## Aktivierung und Gating

Der Hot-Cache ist explizit gated. Das ist wichtig, weil der Fork weiterhin ohne Experiment wie normales llama.cpp laufen soll.

Der neue Pfad ist deaktiviert, wenn:

```text
moe_hot_cache_max_mib == 0
```

Dann passiert beim Laden:

- `llama_moe_hot_cache_init(...)` kehrt ohne Cache-Aufbau zurueck,
- `model.moe_hot_cache` bleibt leer,
- Qwen35Moe verwendet den normalen `build_moe_ffn`-Pfad,
- der Hot-Cache-Graph wird nicht gebaut,
- der Scheduler sieht keine Hot-Cache-Parallelregion.

Der neue Pfad ist aktiv, wenn:

```text
--moe-hot-cache-max-mib <N>
--moe-hot-cache <datei.json>
```

Wenn ein Budget gesetzt ist, aber keine JSON-Datei, ist das absichtlich ein Fehler. Sonst waere unklar, welche Experten gecached werden sollen.

Der Qwen-Hot-Pfad wird pro Layer nur gebaut, wenn:

```text
llama_moe_hot_cache_layer_active(model, il)
```

`true` liefert.

Damit koennen einzelne Layer aktiv sein, andere aber normal laufen.

## CLI- und Env-Schalter

### Hot-Cache-Auswahl

```text
--moe-hot-cache <datei.json>
LLAMA_ARG_MOE_HOT_CACHE=<datei.json>
```

Pfad zur Perf-/Hot-Cache-JSON.

```text
--moe-hot-cache-max-mib <N>
LLAMA_ARG_MOE_HOT_CACHE_MAX_MIB=<N>
```

Maximales Speicherbudget fuer den Hot-Cache in MiB. `0` deaktiviert die Funktion.

### Parallelisierung

```text
LLAMA_MOE_HOT_CACHE_PARALLEL=1
```

Aktiviert Hot/Cold-Parallelisierung im Auto-Modus.

```text
LLAMA_MOE_HOT_CACHE_PARALLEL=force
```

Aktiviert den Force-Modus. Wenn die Region nicht korrekt parallelisiert werden kann, wird nicht still auf seriell zurueckgefallen, sondern ein Fehler erzeugt. Dieser Modus war hilfreich beim Debugging, ist aber fuer normale Tests riskanter.

```text
LLAMA_MOE_HOT_CACHE_PARALLEL=0
```

Deaktiviert die Parallelisierung.

```text
LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS=<N>
```

Mindestanzahl Slots, ab der der Scheduler die Hot/Cold-Parallelregion startet. Default: `64`.

Ziel: Fuer sehr kleine Arbeitspakete ist Thread-/Scheduler-Overhead teurer als parallele Ausfuehrung.

### Decode- und Merge-Optimierungen

Die folgenden Schalter sind als Entwicklungshebel vorhanden. Viele davon sind aktuell default-aktiv, weil sie in den spaeteren Laeufen Performance gebracht haben.

```text
LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING=1
```

Routing im Decode ueber einen CPU-Custom-Op-Pfad.

```text
LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE=1
```

Direkter Merge-Pfad fuer Decode.

```text
LLAMA_MOE_HOT_CACHE_DECODE_STRIDED_SUM_ROWS=1
```

Optimierter Summenpfad fuer Decode-Merge.

```text
LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING=1
```

Padding fuer Hot-Arbeit, damit Graph-/Backend-Erwartungen stabil bleiben.

```text
LLAMA_MOE_HOT_CACHE_SHARED_INPUT_ROW=1
```

Optimierung fuer geteilte Input-Zeilen im Hot-Pfad.

```text
LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM=1
LLAMA_MOE_HOT_CACHE_COLD_PREFIX_WEIGHTED_SUM=1
```

Optimierungen fuer den Cold-Pfad.

```text
LLAMA_MOE_HOT_CACHE_DECODE_REPEAT_HOT_INPUT=1
LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT=1
```

Weitere Decode-spezifische Reduktionen von Gather-/Input-Overhead.

### Perf

```text
--no-perf
```

Deaktiviert die llama.cpp-Perf- und MoE-Perf-Zaehler fuer einen moeglichst sauberen Speed-Test.

```text
LLAMA_MOE_LAYER_PERF=0
```

Deaktiviert die MoE-Layer-Perf-Ausgabe.

```text
LLAMA_MOE_LAYER_PERF_EXPERT_COUNTS=1
```

Aktiviert detaillierte Expert-Counts. Das ist hilfreich, um eine erste Hot-Cache-Liste aus echten Traffic-Daten zu erzeugen. Fuer Speed-Tests sollte es deaktiviert bleiben.

## Dateiuebersicht

### `src/llama-moe-hot-cache.h`

Zentrale Datentypen fuer den Hot-Cache:

- Cache-Konfiguration,
- Layer-Konfiguration,
- Expert-Auswahl,
- Tensor- und Worklist-Layouts,
- Hilfsfunktionen zur Abfrage, ob ein Layer aktiv ist.

Wichtige Worklist-Felder:

```text
HOT_ID
HOT_SRC_SLOT
HOT_TOKEN_ID
HOT_WEIGHT
COLD_ID
COLD_SRC_SLOT
COLD_TOKEN_ID
COLD_WEIGHT
HOT_EXPERT_ID
HOT_COUNT
COLD_COUNT
```

Die Worklist trennt hot und cold Slots. Der Scheduler und die Merge-Logik koennen dadurch beide Pfade getrennt behandeln.

### `src/llama-moe-hot-cache.cpp`

Zustaendig fuer:

- Parsen der JSON-Datei,
- Erkennen unterstuetzter Schemas,
- Sammeln der Expertengroessen,
- Scoring der Experten,
- Auswahl innerhalb des Speicherbudgets,
- Allokation der Hot-Cache-Tensoren,
- Kopieren der ausgewaehlten Experten,
- Aufbau von Mapping- und Maskentabellen.

Unterstuetzte Schemas:

```text
llama.cpp.moe_layer_perf.v1
llama.cpp.moe_layer_opt_perf.v1
```

Die Auswahl gewichtet nicht nur rohe Expert-Counts. Spaetere Optimierungen haben Layer-Wartezeiten und Sticky-Boni fuer bereits ausgewaehlte hot Experten beruecksichtigt, damit das Cache-Set stabiler und praxisnaeher wird.

### `src/llama-moe-hot-cache-graph.cpp`

Enthaelt den Qwen35Moe-spezifischen Hot-Cache-Graph.

Das war ein wichtiger Refactor-Schritt: Der experimentelle Graph-Code wurde aus `src/models/qwen35moe.cpp` herausgezogen, damit das Modellfile upstream-naeher bleibt.

Aufgaben:

- Hot/Cold-FFN fuer Qwen35Moe bauen,
- Decode-spezifische Worklist erstellen,
- CPU-Custom-Ops fuer Routing und Merge einbinden,
- Hot-Branch mit gecachten Experten bauen,
- Cold-Branch mit normalen Experten bauen,
- Branch-Ausgaben zusammenfuehren,
- Scheduler-Parallelregion annotieren,
- lokale `mul_mat_id`-Flags setzen.

Wichtig: Der Code ist absichtlich Qwen35Moe-spezifisch. Er ist kein allgemeiner MoE-Ersatz fuer alle Architekturen.

### `src/models/qwen35moe.cpp`

Dieses File ist nach dem Refactor wieder deutlich kleiner.

Die zentrale Entscheidung ist:

```text
Wenn der Layer einen aktiven Hot-Cache hat:
    build_layer_ffn_hot(...)
Sonst:
    normaler build_moe_ffn(...)
```

Damit bleibt der normale Modellpfad erhalten.

### `src/llama-moe-hot-cache-perf.h`

Deklariert die MoE-Perf-Schnittstellen.

Wichtig ist vor allem:

```text
llama_moe_layer_perf_json(ctx)
```

Diese Funktion ist auch ueber die oeffentliche API in `include/llama.h` erreichbar.

### `src/llama-moe-hot-cache-perf.cpp`

Enthaelt die MoE-spezifische Perf-Sammlung und JSON-Ausgabe.

Vor dem Refactor lag relevante Logik in `llama-context.cpp`. Sie wurde ausgelagert, damit der Kernkontext upstream-naeher bleibt.

Erfasste Daten:

- globale Summary,
- Layer-Calls,
- Hot-Slot-Ratio,
- Routing-Zeit,
- Worklist-Zeit,
- Merge-Zeit,
- Hot-Branch-Zeit,
- Cold-Branch-Zeit,
- Hot-/Cold-Matmul-Zeit,
- Gather-/Scatter-Zeiten,
- Parallelregion-Zeiten,
- Join-Wartezeit,
- Overlap,
- Launch-/Fallback-Counts,
- optional Expert-Counts.

Wenn Perf deaktiviert ist, liefert die JSON-Funktion absichtlich nur einen kleinen Disabled-Block:

```json
{
  "enabled": false,
  "schema": "llama.cpp.moe_layer_opt_perf.v1",
  "layers": []
}
```

### `src/llama-context.cpp`

Nur noch kleine Hooks fuer:

- Perf-Callback setzen,
- Scheduler-Perf aktivieren,
- Graph-Compute wie vorher ausfuehren.

Wenn `--no-perf` aktiv ist, wird der MoE-Perf-Pfad nicht gesetzt.

### `src/llama.cpp`

Initialisiert den Hot-Cache beim Modell-Laden:

```text
llama_moe_hot_cache_init(*model, params)
```

Die Funktion ist durch `moe_hot_cache_max_mib` gated.

### `src/llama-model.cpp` und verwandte Param-Dateien

Erweitern die Modellparameter um:

```text
moe_hot_cache_path
moe_hot_cache_max_mib
```

Das Modell besitzt ausserdem den Hot-Cache-Zustand:

```text
std::unique_ptr<llama_moe_hot_cache> moe_hot_cache
```

### `src/llama-graph.cpp` und `src/llama-graph.h`

Diese Dateien wurden wieder upstream-naeher gemacht.

Fruehere Hot-Cache-spezifische Erweiterungen wie:

- spezieller `build_moe_ffn_with_ids`,
- `llm_mul_mat_id_flags`,
- Hot-Cache-spezialisiertes `build_lora_mm_id`-Verhalten,

wurden aus dem allgemeinen Graph-Code herausgezogen.

Das reduziert Konflikte bei upstream-Updates.

### `ggml/include/ggml-backend-moe-hot-cache.h`

Neuer experimenteller Header fuer die Scheduler-API des Hot-Cache.

Der Grund fuer den separaten Header: `ggml-backend.h` soll moeglichst wenig experimentelle API enthalten.

### `ggml/src/ggml-backend-moe-hot-cache.inc`

Private Implementierung der Scheduler-Erweiterung.

Dieses `.inc` wird aus `ggml-backend.cpp` eingebunden. Dadurch kann der Code weiterhin auf interne Scheduler-Strukturen zugreifen, ohne die komplette Implementierung direkt in `ggml-backend.cpp` stehen zu lassen.

Aufgaben:

- Parallelregionen registrieren,
- Split-Boundaries erzwingen,
- Region validieren,
- Hot- und Cold-Splits identifizieren,
- CPU-Worker fuer Cold-Pfad starten,
- Hot-Pfad auf dem normalen Thread/GPU-Backend laufen lassen,
- beide Pfade joinen,
- Fehler/Fallbacks erfassen,
- Scheduler-Perf-Daten sammeln.

### `ggml/src/ggml-backend.cpp`

Nur noch kleine Integrationspunkte:

- Scheduler-State besitzt einen Hot-Cache-Zustandspointer,
- Init/Free/Reset hooken den Zustand,
- Split-Boundaries fragen die Hot-Cache-Logik,
- das `.inc` wird eingebunden.

Das ist absichtlich so gehalten, damit upstream-Konflikte kleiner bleiben.

### `ggml/src/ggml-cpu/ggml-cpu.c`

CPU-Backend-Anpassungen fuer die Hot-Cache-MoE-Arbeit.

Relevante Punkte:

- `MUL_MAT_ID` kann ueber `op_params` fuer Hot-Cache-Layouts gesteuert werden,
- negative IDs koennen erlaubt werden,
- Zeroing kann uebersprungen werden, wenn ein Pfad garantiert keine Ausgabe schreiben muss,
- bestimmte Decode-Layouts mit geteilten Input-Zeilen werden unterstuetzt.

Diese Anpassung ist eine der Stellen, die nicht komplett ausserhalb des Kerns liegen kann.

### `ggml/src/ggml-cuda/ggml-cuda.cu`

CUDA-Backend-Anpassungen fuer `MUL_MAT_ID`-Varianten, die im Hot-Pfad gebraucht werden.

Relevante Punkte:

- Hot-Cache-Flags aus `op_params`,
- Unterstuetzung spezieller ID-Layouts,
- Vermeidung unnoetiger Arbeit bei bekannten Decode-Faellen.

Auch diese Anpassung ist ein Kernhook und deshalb eine moegliche Konfliktstelle bei upstream-Updates.

### `tests/test-moe-hot-cache.cpp`

Unit Tests fuer:

- JSON-Auswahl,
- Budget-/Layer-Verhalten,
- Worklist-/Mapping-Grundlagen,
- Gating-Faelle.

Bei weiteren Refactors sollte dieses Testfile erweitert werden, bevor Optimierungen in den Decode-Pfad gehen.

## Laufzeitablauf

### 1. Startparameter werden gelesen

Die CLI/Common-Params lesen Hot-Cache-Argumente und Perf-Schalter.

Wichtig:

```text
--moe-hot-cache-max-mib 0
```

ist der Default und bedeutet: kein Hot-Cache.

### 2. Modell wird geladen

Wenn `moe_hot_cache_max_mib > 0` ist, wird beim Modell-Laden der Hot-Cache initialisiert.

Der Cache braucht:

- Modell-Metadaten,
- Layer-/Expert-Groessen,
- Perf-JSON,
- Speicherbudget,
- Backend-Informationen.

Wenn keine GPU-Backend-Konfiguration passt, wird der Hot-Cache nicht sinnvoll nutzbar. Der aktuelle experimentelle Parallelpfad erwartet hot auf CUDA und cold auf CPU.

### 3. Hot-Experten werden ausgewaehlt

Die JSON-Datei enthaelt Layer- und Expertendaten. Daraus berechnet der Cache eine sortierte Auswahl.

Das Ziel ist nicht einfach "die meistgenutzten Experten". Entscheidend ist:

- wie oft ein Experte benutzt wird,
- in welchem Layer er liegt,
- ob dieser Layer Wartezeit verursacht,
- wie teuer der Experte im Speicher ist,
- ob er zum bisherigen stabilen Hot-Set passt.

Das Budget limitiert die Menge.

### 4. Hot-Cache-Tensoren werden gebaut

Ausgewaehlte Experten werden in Hot-Cache-Tensoren kopiert.

Dadurch kann der Hot-Pfad spaeter mit dichterem Layout arbeiten und muss nicht jedes Mal ueber das normale Expertenlayout gehen.

### 5. Graph wird gebaut

Fuer jeden Qwen35Moe-Layer:

```text
if hot cache active for layer:
    build_layer_ffn_hot(...)
else:
    build_layer_ffn(...)
```

Der normale Pfad bleibt also pro Layer verfuegbar.

### 6. Worklist wird berechnet

Im Decode-Fall ist `n_tokens == 1`. Genau dieser Pfad wurde stark optimiert.

Die Worklist trennt:

- hot Slots,
- cold Slots,
- Gewichte,
- Quellslot,
- Token-ID,
- Expert-ID.

Diese Trennung ist die Grundlage fuer parallele Ausfuehrung.

### 7. Hot-Branch und Cold-Branch werden gebaut

Hot-Branch:

- nutzt gecachte Experten,
- ist fuer CUDA/GPU gedacht,
- soll moeglichst wenig CPU-Overhead verursachen.

Cold-Branch:

- nutzt normale Experten,
- ist fuer CPU gedacht,
- laeuft parallel zur GPU-Arbeit, wenn genug Slots vorhanden sind.

### 8. Scheduler-Region wird annotiert

Der Graph markiert den Bereich als Hot-Cache-Parallelregion:

```text
ggml_backend_sched_moe_hot_cache_parallel_region(...)
```

Die Annotation enthaelt:

- Layer-ID,
- Hot-Count-Tensor,
- Cold-Count-Tensor,
- Start-/End-Tensoren,
- Output-/Join-Tensoren.

### 9. Scheduler validiert die Region

Der Scheduler parallelisiert nur, wenn die Region sauber erkennbar ist.

Validierungen:

- Region ist vollstaendig,
- Count-Tensoren liegen im erwarteten Prefix,
- Split-Reihenfolge ist gueltig,
- Hot- und Cold-Bereiche sind getrennt,
- Hot liegt auf CUDA,
- Cold liegt auf CPU,
- Backends sind nicht identisch,
- Count-Readback funktioniert,
- Mindestslotzahl ist erreicht.

Im Auto-Modus kann der Scheduler auf seriell zurueckfallen. Im Force-Modus ist ein Fehler sichtbar.

Ein frueher Fehler war:

```text
region split order is not hot-then-cold-then-join
```

Der Scheduler erwartete damals eine engere Split-Reihenfolge als der Graph erzeugte. Spaetere Refactors haben die Erkennung und Validierung robuster gemacht.

### 10. Parallel Compute

Die Threads sind CPU-Threads.

Wichtig: Die GPU fuehrt keine "CPU-Threads" aus. Der Ablauf ist:

- Hauptthread startet/verwaltet den Hot-Pfad auf dem CUDA-Backend,
- ein CPU-Worker verarbeitet den Cold-Pfad,
- CUDA arbeitet asynchron auf der GPU,
- CPU und GPU ueberlappen zeitlich,
- am Join wird synchronisiert.

Der Thread-Start selbst ist Overhead. Deshalb gibt es `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS`.

### 11. Merge

Nach Hot und Cold werden die Ergebnisse in den normalen FFN-Ausgang zusammengefuehrt.

Der Merge war ein grosser Optimierungshebel, weil er pro Decode-Schritt und pro Layer laeuft. Die spaeteren direkten Decode-Merge-Pfade reduzieren unnoetige Tensorarbeit.

## Scheduler-Fallbacks

Die Perf-Daten koennen Fallback-Gruende enthalten.

Typische Gruende:

| Grund | Bedeutung |
| --- | --- |
| `incomplete` | Region war nicht vollstaendig annotiert |
| `count_not_prefix` | Count-Tensoren lagen nicht im erwarteten Prefix |
| `bad_split_order` | Splits waren nicht in gueltiger Hot/Cold/Join-Ordnung |
| `same_backend` | Hot und Cold lagen auf demselben Backend |
| `hot_spans_backends` | Hot-Bereich war nicht eindeutig einem Backend zugeordnet |
| `cold_spans_backends` | Cold-Bereich war nicht eindeutig einem Backend zugeordnet |
| `hot_not_cuda` | Hot lag nicht auf CUDA |
| `cold_not_cpu` | Cold lag nicht auf CPU |
| `count_readback` | Count-Readback ist fehlgeschlagen |
| `threshold` | Zu wenige Slots fuer Parallelisierung |
| `zero_output` | Ausgabe musste wegen leerem Pfad genullt werden |

Ein guter Lauf sollte wenige oder keine unerwarteten Fallbacks haben. Threshold-Fallbacks koennen normal sein, wenn sehr kleine Arbeitspakete nicht parallelisiert werden.

## Perf-JSON

Die Perf-Ausgabe wurde bewusst zurueckgebaut. Frueher war sie staerker fuer Visualisierung von Layer-/Expert-Hitmaps gedacht. Fuer Optimierung brauchen wir andere Daten:

- wo Zeit verloren geht,
- wie viel hot/cold Arbeit entsteht,
- wie gut CPU und GPU ueberlappen,
- wie teuer Routing und Merge sind,
- ob Scheduler-Fallbacks passieren,
- ob die Hot-Liste wirklich trifft.

Aktuelles Schema:

```text
llama.cpp.moe_layer_opt_perf.v1
```

Wichtige Felder:

```text
summary.layer_calls
summary.hot_slot_ratio
summary.total_moe_us_per_call
summary.routing_us_per_call
summary.worklist_us_per_call
summary.merge_us_per_call
summary.parallel_region_us_per_call
summary.parallel_hot_us_per_call
summary.parallel_cold_us_per_call
summary.parallel_join_wait_us_per_call
summary.parallel_overlap_us_per_call
summary.parallel_launches
summary.parallel_fallbacks
layers[]
```

Optional mit:

```text
LLAMA_MOE_LAYER_PERF_EXPERT_COUNTS=1
```

kommen detaillierte Expertenzaehler dazu. Diese sind fuer die initiale Cache-Erzeugung nuetzlich, aber fuer reine Speed-Laeufe teuer und deshalb standardmaessig aus.

## Wie man eine erste Hot-Cache-Liste erzeugt

Wenn noch keine Hot-Cache-JSON existiert, startet man ohne Hot-Cache, aber mit Expert-Counts:

```bash
LLAMA_MOE_LAYER_PERF_EXPERT_COUNTS=1 \
./build/bin/llama-server \
  --perf \
  <normale modell- und server-argumente>
```

Dann laesst man repraesentative Prompts laufen und liest die MoE-Perf-JSON aus dem Server, zum Beispiel ueber den vorhandenen `/moe-layer-perf`-Pfad.

Diese JSON kann danach als Input fuer den Hot-Cache verwendet werden:

```bash
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib <budget> \
  <normale modell- und server-argumente>
```

Fuer finale Durchsatzmessungen:

```bash
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --no-perf \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib <budget> \
  <normale modell- und server-argumente>
```

## Warum `--no-perf` wichtig ist

Die Perf-Zaehler sind nicht kostenlos.

Auch wenn die JSON-Ausgabe selten abgefragt wird, muessen waehrend Decode Messpunkte, Counter und teils Scheduler-Daten aktualisiert werden. Bei sehr kleinen Decode-Schritten ist dieser Overhead sichtbar.

Deshalb gibt es zwei Modi:

Entwicklungsmodus:

```text
--perf
LLAMA_MOE_LAYER_PERF=1
```

Ziel: verstehen, wo Zeit verloren geht.

Speed-Modus:

```text
--no-perf
LLAMA_MOE_LAYER_PERF=0
```

Ziel: reale Geschwindigkeit ohne Messlast.

## Warum der erste parallele PoC langsamer war

Der erste funktionierende parallele Pfad war absichtlich noch nicht optimal. Er zeigte, dass die Aufteilung technisch funktioniert, aber er hatte viel Overhead:

- Routing war zu teuer,
- Merge war zu teuer,
- Scheduler-Start/Join war zu teuer,
- zu kleine Arbeitspakete wurden parallelisiert,
- Hot-Liste passte nicht immer zum Prompt,
- detaillierte Perf-Zaehler liefen mit,
- Cold-Pfad war noch nicht stark optimiert.

Dadurch konnte ein einfacher Prompt langsamer werden, obwohl ein Programmiertask bereits besser passte. Das war erwartbar, weil die Hot-Liste aus Programmiertraffic entstanden war.

## Warum spaetere Laeufe schneller wurden

Die spaeteren Verbesserungen kamen aus mehreren Hebeln.

### 1. Decode-Routing reduziert

Der Decode-Pfad hat nur einen Token. Allgemeine MoE-Graphlogik fuer viele Tokens ist hier zu teuer.

Deshalb wurde ein spezieller Decode-Worklist-Pfad eingefuehrt. Er reduziert:

- unnoetige Tensoroperationen,
- unnoetige Kopien,
- unnoetige Sortier-/Mapping-Arbeit.

### 2. Merge reduziert

Der Merge passiert nach jedem Hot/Cold-Branch. Ein ineffizienter Merge frisst den Vorteil der Parallelisierung.

Die Optimierungen:

- direkter Decode-Merge,
- strided sum rows,
- weniger Zeroing,
- bessere Nutzung bekannter Decode-Layouts.

### 3. Parallelregion nur bei genug Arbeit

Der CPU-Worker muss gestartet oder aufgeweckt werden. Das lohnt sich nicht fuer wenige Slots.

`LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS` vermeidet kleine, teure Parallelregionen.

### 4. Cold-Pfad optimiert

Da die Hot-Hit-Rate nicht 100 Prozent erreichen kann, bleibt der Cold-Pfad wichtig.

Optimierungen im Cold-Pfad waren besonders wertvoll, weil der Join sonst auf CPU-Arbeit wartet.

### 5. Perf-Pfad minimiert

Die Perf-Ausgabe wurde von Visualisierungsdaten auf Optimierungsdaten umgestellt.

Dadurch bleibt sie im Entwicklungsmodus nuetzlich, verursacht aber weniger Last. Fuer finale Messungen kann sie komplett deaktiviert werden.

### 6. Refactor zur Isolation

Experimenteller Code wurde in eigene Dateien verschoben. Das verbessert nicht direkt tk/s, senkt aber das Risiko, dass Performance-Optimierungen spaeter schwer wartbar werden.

## Refactor-Historie

Die folgenden Commits markieren den groben Verlauf seit dem Einstiegspunkt `0641adb9e08da0a675058bc39a8c928a7f8d6ad0`.

| Commit | Zweck |
| --- | --- |
| `5ad16adeb` | Erste Inferenztests, funktional aber langsamer |
| `c26ec8556` | Laufender PoC |
| `61aaa4c75` | Mehr Metriken |
| `338c5ee05` | Kaputter Parallelisierungsversuch |
| `2a795fc3a` | Revert des kaputten Versuchs |
| `dac2548ac` | CPU- und GPU-Arbeit parallel berechnet |
| `39a942ab1` | Scheduler optimiert |
| `50284e9b8` | Speed auf gutem Pfad |
| `220b44da3` | Andere Methode fuer Hot-Expert-Auswahl |
| `5759b13aa` | Gewichte angepasst |
| `03b23a2f6` | Routing verbessert |
| `9987d8266` | Perf-Pfad minimiert |
| `19c309c84` | Perf deaktivierbar gemacht |
| `d7305c2d0` | Weitere Performance-Verbesserungen |
| `a47d68214` | Cold-Pfad schneller gemacht |
| `4b47c82c5` | Neuer Stand um ca. 28 tk/s |
| `e91fb763a` | Refactor Teil 1: Qwen-Hot-Graph separiert |
| `3de19d8ae` | Refactor Teil 2: allgemeiner Graph upstream-naeher |
| `106e407ea` | Refactor Teil 3: Perf-Code separiert |
| `0c3b4a58e` | Refactor Teil 4: Scheduler-Implementierung separiert |
| `e3ace0d3b` | Refactor Teil 5: experimentelle Scheduler-API separiert |

## Details zu den Refactor-Teilen

### Teil 1: Qwen-Hot-Graph separiert

Vorher lag viel Hot-Cache-Graphlogik direkt in `src/models/qwen35moe.cpp`.

Problem:

- Modellfile wurde gross,
- upstream-Konflikte waeren wahrscheinlich,
- schwer zu erkennen, welcher Code experimentell ist.

Loesung:

- `src/llama-moe-hot-cache-graph.cpp` angelegt,
- `qwen35moe.cpp` auf kleinen Gate reduziert,
- normaler Qwen-Pfad bleibt sichtbar und unveraendert.

### Teil 2: Allgemeiner Graph upstream-naeher

Vorher waren Hot-Cache-spezifische Flags und Hilfsfunktionen in `llama-graph.*`.

Problem:

- allgemeiner Graph-Code wurde mit experimenteller Semantik vermischt,
- andere Modelle koennten indirekt betroffen wirken,
- Upstream-Merges werden schwerer.

Loesung:

- Hot-Cache-spezifisches `build_moe_ffn_with_ids` entfernt oder verschoben,
- `build_lora_mm_id` wieder allgemeiner gehalten,
- `mul_mat_id`-Flags lokal im Hot-Cache-Graph verwaltet.

### Teil 3: Perf-Code separiert

Vorher lag MoE-Perf-Logik im Kontextcode.

Problem:

- `llama-context.cpp` wurde unuebersichtlich,
- Perf war schwer getrennt zu deaktivieren,
- Konfliktflaeche zu upstream war gross.

Loesung:

- `src/llama-moe-hot-cache-perf.h`
- `src/llama-moe-hot-cache-perf.cpp`
- Kontext nur noch mit kleinen Hooks.

### Teil 4: Scheduler-Implementierung separiert

Vorher lag viel Hot-Cache-Schedulerlogik in `ggml-backend.cpp`.

Problem:

- `ggml-backend.cpp` ist upstream-kritisch,
- grosse lokale Aenderungen erzeugen Merge-Konflikte,
- experimenteller Scheduler-Code war schwer zu erkennen.

Loesung:

- `ggml/src/ggml-backend-moe-hot-cache.inc`,
- kleiner Zustandspointer im Scheduler,
- kleine Init/Free/Reset/Split-Hooks im Kern.

### Teil 5: Experimentelle Scheduler-API separiert

Vorher lag Hot-Cache-API in `ggml-backend.h`.

Problem:

- oeffentlicher Kernheader wurde mit experimenteller API erweitert,
- schwerer upstream-nahe zu bleiben.

Loesung:

- `ggml/include/ggml-backend-moe-hot-cache.h`,
- Hot-Cache-Code inkludiert diesen Header explizit,
- allgemeiner Backend-Header bleibt sauberer.

## Was weiterhin Kernhooks braucht

Komplette Isolation ist nicht moeglich, solange wir echte Parallelisierung im bestehenden Graph/Scheduler wollen.

Folgende Stellen bleiben bewusst Kernhooks:

- Scheduler-State in `ggml-backend.cpp`,
- Split-Boundary-Erkennung,
- Include der privaten Scheduler-Erweiterung,
- CPU- und CUDA-`MUL_MAT_ID`-Semantik fuer Hot-Cache-Layouts,
- Modellparameter fuer Hot-Cache-CLI,
- kleine Hook-Stellen im Kontext fuer Perf.

Diese Stellen sollten klein bleiben. Neue Hot-Cache-Logik sollte bevorzugt in:

```text
src/llama-moe-hot-cache*.*
ggml/src/ggml-backend-moe-hot-cache.inc
ggml/include/ggml-backend-moe-hot-cache.h
```

landen.

## Update-Strategie fuer upstream llama.cpp

Beim Rebase oder Merge von upstream:

1. zuerst Konflikte in generischen Dateien klein halten,
2. pruefen, ob `ggml-backend.cpp` Scheduler-Strukturen geaendert hat,
3. pruefen, ob `ggml-cpu.c` oder `ggml-cuda.cu` `MUL_MAT_ID` geaendert haben,
4. Qwen35Moe-Modelldatei nur auf den kleinen Gate vergleichen,
5. danach Hot-Cache-spezifische Dateien separat testen.

Besonders konfliktanfaellig:

```text
ggml/src/ggml-backend.cpp
ggml/src/ggml-cpu/ggml-cpu.c
ggml/src/ggml-cuda/ggml-cuda.cu
src/llama.cpp
src/llama-context.cpp
src/llama-model.cpp
src/models/qwen35moe.cpp
```

Weniger konfliktanfaellig:

```text
src/llama-moe-hot-cache.cpp
src/llama-moe-hot-cache.h
src/llama-moe-hot-cache-graph.cpp
src/llama-moe-hot-cache-perf.cpp
src/llama-moe-hot-cache-perf.h
ggml/src/ggml-backend-moe-hot-cache.inc
ggml/include/ggml-backend-moe-hot-cache.h
tests/test-moe-hot-cache.cpp
```

## Entwicklungsworkflow

### Build

Der bevorzugte Build-Befehl in diesem Projekt ist:

```bash
cmake --build build -j8
```

Fuer schnellere Pruefung einzelner Targets:

```bash
cmake --build build -j8 --target llama
cmake --build build -j8 --target test-moe-hot-cache
```

### Unit Test

```bash
./build/bin/test-moe-hot-cache
```

### Server ohne Hot-Cache

Dieser Modus sollte wie normales llama.cpp laufen:

```bash
./build/bin/llama-server <normale argumente>
```

Wichtig: Kein `--moe-hot-cache-max-mib` setzen.

### Server mit Hot-Cache und Perf

```bash
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --perf \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib <budget> \
  <normale argumente>
```

### Server mit Hot-Cache ohne Perf

```bash
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --no-perf \
  --moe-hot-cache performance.json \
  --moe-hot-cache-max-mib <budget> \
  <normale argumente>
```

### Optionaler CLI-Test

Der CLI-Test war fuer schnelle Reproduktion hilfreich, ist aber in der konkreten Umgebung nicht immer stabil gewesen.

```bash
/home/adrian/llama.exp/build/bin/llama-cli \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q6_K_XL \
  -dev CUDA0 \
  --ctx-size 32000 \
  -p "Antworte mit Hallo" \
  -st
```

Der produktivere Testpfad ist aktuell der Server.

## Interpretation der Perf-Daten

Beim Lesen einer neuen `performance*.json` sind diese Fragen am wichtigsten:

1. Wie hoch ist `hot_slot_ratio`?
2. Gibt es unerwartete Scheduler-Fallbacks?
3. Ist `parallel_join_wait_us_per_call` hoch?
4. Ist der Cold-Pfad zu langsam?
5. Ist der Merge wieder sichtbar teuer?
6. Ist Routing/Worklist im Decode sichtbar?
7. Sind Hot- und Cold-Branch wirklich ueberlappt?
8. Hat `--no-perf` eine relevante Differenz?

Wenn `hot_slot_ratio` bei ca. 70 Prozent gedeckelt ist, kann der naechste grosse Gewinn nur durch weniger Overhead und schnelleren Cold-Pfad kommen. Mehr Hot-Cache allein hilft dann nur begrenzt.

## Was passiert bei mehr `n_expert_used`

Wenn `n_expert_used` von 8 auf 16 steigt, entstehen pro Token mehr Expert-Slots.

Erwartete Effekte:

- mehr Gesamtarbeit,
- mehr Routing- und Worklist-Arbeit,
- mehr Merge-Arbeit,
- potenziell mehr Parallelisierungsmoeglichkeit,
- aber auch mehr Cold-Arbeit, wenn der Hot-Cache nicht entsprechend trifft,
- Hot-Cache-Budget kann schneller knapp werden.

Mehr Experten pro Token ist deshalb kein automatischer Speed-Gewinn. Es kann die Qualitaet oder Modellcharakteristik aendern, aber fuer diesen Optimierungspfad muss die Hot-Hit-Rate und der Cold-Overhead neu gemessen werden.

## Bekannte Grenzen

### Statischer Cache

Der Hot-Cache ist aktuell statisch. Er wird beim Start aus einer JSON-Datei gebaut.

Ein dynamisches Update nach jedem Inferenzlauf ist nicht implementiert.

Ein dynamischer Cache waere moeglich, aber er waere deutlich komplexer:

- neue Expertenauswahl waehrend Laufzeit,
- sichere Tensor-Neuallokation oder doppelter Cache,
- Synchronisierung mit laufenden Graphen,
- GPU/CPU-Kopien im Hintergrund,
- Schutz vor instabilen Hot-Sets.

### CUDA/CPU-Annahme

Der Parallelpfad ist praktisch fuer:

```text
hot: CUDA
cold: CPU
```

Andere Backends sind nicht das Ziel dieses Codes. Fuer Vulkan, Metal oder mehrere CUDA-GPUs muesste die Scheduler-Validierung und Graph-Zuweisung erweitert werden.

### Qwen35Moe-Spezifik

Der Hot-Graph ist fuer Qwen35Moe gebaut. Andere MoE-Modelle nutzen den normalen Pfad, solange sie nicht explizit angebunden werden.

### Perf-Daten sind workload-abhaengig

Eine Hot-Liste aus Programmiertasks kann einfache Prompts schlechter treffen. Deshalb sollten Hot-Cache-JSONs aus repraesentativem Traffic entstehen.

## Typische Fehlerbilder

### Assertion in `ggml_view_1d`

Frueherer Fehler:

```text
GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src)) failed
```

Ursache war ein fehlerhaftes View-/Worklist-Layout im ersten Inferenzpfad. Der Fix lag im korrekten Dimensionieren und Trennen der Hot-/Cold-Worklistbereiche.

### Scheduler meldet falsche Split-Reihenfolge

Fehler:

```text
forced MoE hot-cache parallel region failed for layer 0:
region split order is not hot-then-cold-then-join
```

Ursache war, dass Graph-Splits nicht der damals erwarteten Regionserkennung entsprachen. Die spaetere Scheduler-Erkennung wurde robuster und der Graph wurde sauberer annotiert.

### Server haengt im Warmup

Moegliche Ursachen:

- Force-Modus mit Region, die nicht validiert,
- zu aggressive Parallelregion,
- Count-Readback/Synchronisation,
- Warmup laeuft mit anderer Slotverteilung als Decode,
- Perf-/Expert-Counts verursachen unerwartete Last.

Debug-Reihenfolge:

1. ohne `LLAMA_MOE_HOT_CACHE_PARALLEL` testen,
2. Auto-Modus statt `force`,
3. `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS` erhoehen,
4. `--no-perf` testen,
5. Perf-JSON auf Fallbacks pruefen.

## Regeln fuer weitere Optimierungen

### 1. Erst messen, dann aendern

Jede neue Optimierung sollte mindestens diese Werte vor/nachher vergleichen:

- tk/s am Ende eines laengeren Laufs,
- `hot_slot_ratio`,
- `parallel_join_wait_us_per_call`,
- `parallel_overlap_us_per_call`,
- `cold_branch_us_per_call`,
- `merge_us_per_call`,
- `routing_us_per_call`,
- Fallback-Counts.

### 2. Normale Pfade nicht anfassen, wenn es vermeidbar ist

Wenn eine Aenderung nur Hot-Cache betrifft, gehoert sie in Hot-Cache-Dateien.

### 3. Kernhooks klein halten

Wenn `ggml-backend.cpp`, `ggml-cpu.c` oder `ggml-cuda.cu` geaendert werden, sollte die Aenderung minimal und gut begruendet sein.

### 4. Decode ist wichtiger als Prefill

Das Hauptziel ist TG/Decode. Prefill darf nicht kaputtgehen, aber der groesste Nutzen liegt im Ein-Token-Pfad.

### 5. Force-Modus nur zum Debuggen

`LLAMA_MOE_HOT_CACHE_PARALLEL=force` ist gut, um Validierungsfehler sofort zu sehen. Fuer normale Performance-Laeufe ist Auto-Modus robuster.

## Offene Optimierungshebel

Die naechsten Hebel liegen wahrscheinlich in diesen Bereichen:

### Cold-Pfad weiter reduzieren

Wenn die Hot-Hit-Rate bei ca. 70 Prozent gedeckelt ist, bleiben 30 Prozent Cold-Arbeit. Jede Reduktion im Cold-Pfad wirkt direkt auf Join-Wartezeit.

Moegliche Richtungen:

- weniger Gather im Cold-Pfad,
- bessere Batch-/Slot-Kompaktion,
- weniger Gewichtsanwendung im separaten Schritt,
- weitere Decode-Spezialisierung.

### Join-Wartezeit reduzieren

Wenn die GPU frueh fertig ist und auf CPU wartet, muss die Cold-Arbeit kleiner oder frueher gestartet werden.

Moegliche Richtungen:

- Count-/Worklist-Erstellung so legen, dass Cold frueher starten kann,
- Cold-Branch kleiner schneiden,
- Scheduler-Startkosten weiter amortisieren.

### Hot-Cache-Auswahl verbessern

Wenn mehr Daten vorliegen, kann die Auswahl besser werden.

Moegliche Richtungen:

- Layer-Wartezeit staerker gewichten,
- Prompt-Klassen trennen,
- mehrere Cache-Profile,
- Hot-Set stabilisieren, damit nicht einzelne Ausreisser dominieren.

### Dynamischer Cache

Langfristig koennte ein dynamischer Cache helfen, wenn Workloads stark wechseln. Kurzfristig ist er aber komplexer als weitere Overhead-Reduktion.

### Backend-spezifische Matmul-ID-Pfade pruefen

Ein Teil des Overheads liegt in `MUL_MAT_ID`-Spezialfaellen. Hier koennte noch Performance liegen, aber das ist kernnah und konflikttraechtig.

## Mindest-Checkliste fuer neue Aenderungen

Vor einem Commit:

```bash
cmake --build build -j8 --target llama
cmake --build build -j8 --target test-moe-hot-cache
./build/bin/test-moe-hot-cache
```

Wenn Runtime betroffen ist:

1. Server ohne Hot-Cache starten und einfache Inferenz testen.
2. Server mit Hot-Cache und Perf starten.
3. `/moe-layer-perf` pruefen.
4. Server mit Hot-Cache und `--no-perf` starten.
5. tk/s mit vorherigem Lauf vergleichen.

## Zusammenfassung fuer zukuenftige Entwickler

Dieser Fork beschleunigt Qwen3.x-MoE-Decode, indem er haeufige Experten in einen Hot-Cache legt und hot GPU-Arbeit parallel zu cold CPU-Arbeit ausfuehrt.

Der normale llama.cpp-Pfad bleibt aktiv, solange kein Hot-Cache-Budget gesetzt ist.

Der experimentelle Code wurde nachtraeglich weitgehend aus generischen Dateien herausgezogen:

- Graph-Speziallogik in `llama-moe-hot-cache-graph.cpp`,
- Cache-Aufbau in `llama-moe-hot-cache.cpp`,
- Perf in `llama-moe-hot-cache-perf.cpp`,
- Scheduler-Erweiterung in `ggml-backend-moe-hot-cache.inc`,
- experimentelle Scheduler-API in `ggml-backend-moe-hot-cache.h`.

Einige Kernhooks bleiben noetig, besonders im Scheduler und bei `MUL_MAT_ID`. Diese Hooks sollten klein bleiben und bei upstream-Updates besonders sorgfaeltig geprueft werden.

Die Performance-Entwicklung ging von ca. 16 bis 17 tk/s in fruehen Programmiertask-Laeufen auf ca. 28.09 tk/s im letzten genannten Lauf. Der wichtigste weitere Hebel ist nicht mehr "einfach mehr Hot-Cache", sondern weniger Overhead und ein schnellerer Cold-Pfad, weil die erreichbare Hot-Hit-Rate realistisch bei etwa 70 Prozent liegt.
