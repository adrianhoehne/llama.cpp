# MoE Hot Cache Runtime Switches

Diese Datei fasst die MoE-Hot-Cache-Umgebungsvariablen zusammen, die in diesem Fork relevant sind.

Hinweise:

- `LLAMA_ARG_*` sind die normalen llama.cpp-Env-Aliase fuer CLI-Argumente.
- In `model_config.ini` wird der Argumentname ohne fuehrendes `--` verwendet, zum Beispiel `moe-hot-cache-max-mib = -1`.
- Die reinen `LLAMA_MOE_*`-Schalter sind Entwicklungs- und Laufzeithebel. Fuer sie gibt es aktuell kein CLI-/INI-Argument, ausser es ist in der Tabelle explizit genannt.
- Boolean-Schalter mit Default `on` akzeptieren praktisch: unset/leer/on/true/1 = an, `0`/`off`/`false` = aus.

## Argumentgestuetzte Optionen

| Env-Variable | Passendes Argument / INI-Key | Optionen | Default | Wirkung |
|---|---|---|---|---|
| `LLAMA_ARG_MOE_HOT_CACHE` | `--moe-hot-cache FNAME` / `moe-hot-cache = FNAME` | Pfad zu JSON-Datei | unset | Gibt die `/moe-layer-perf`-JSON an, aus der der initiale Hot Cache gebaut wird. Erforderlich, wenn `--moe-hot-cache-max-mib` ungleich `0` ist. |
| `LLAMA_ARG_MOE_HOT_CACHE_MAX_MIB` | `--moe-hot-cache-max-mib N` / `moe-hot-cache-max-mib = N` | `0`, `>0`, `-1` | `0` | `0` deaktiviert den Hot Cache. Positive Werte setzen ein fixes MiB-Budget. `-1` auto-sized aus dem nach Model/KV verbleibenden VRAM und benoetigt einen expliziten `--ctx-size`. |
| `LLAMA_ARG_MOE_HOT_CACHE_AUTO_RESERVE_MIB` | `--moe-hot-cache-auto-reserve-mib N` / `moe-hot-cache-auto-reserve-mib = N` | Ganzzahl `>= 0` | `1024` | Wird nur mit `--moe-hot-cache-max-mib -1` genutzt. Laesst nach der Auto-Budget-Berechnung noch N MiB fuer Warmup, Compute-Buffer und CUDA-Transienten frei. |
| `LLAMA_ARG_MOE_HOT_CACHE_UPDATE_RATE` | `--moe-hot-cache-update-rate N` / `moe-hot-cache-update-rate = N` | Float `0.0` bis `1.0` | `0.0` | Tauscht nach abgeschlossenen Server-Laeufen bis zu N Anteil der aktuellen Hot-Cache-Eintraege gegen bessere Kandidaten aus. Benoetigt einen aktiven Hot Cache. |
| `LLAMA_ARG_MOE_HOT_CACHE_QWEN_LAYER_CURVE`; intern auch `LLAMA_MOE_HOT_CACHE_QWEN_LAYER_CURVE` | `--moe-hot-cache-qwen-layer-curve N` / `moe-hot-cache-qwen-layer-curve = N` | Float `0.0` bis `1.0` | `0.5` | Nur Qwen35Moe/Qwen3.6-MoE. Gewichtet Layer mit hoeherem Zeitdruck bei initialer Auswahl und dynamischem Update staerker. `0.0` = keine Layer-Druck-Gewichtung, `1.0` = aggressive Layer-Druck-Gewichtung. Das Argument setzt intern die `LLAMA_MOE_HOT_CACHE_QWEN_LAYER_CURVE`-Env. |
| `LLAMA_ARG_MOE_HOT_CACHE_WEIGHTING`; intern auch `LLAMA_MOE_HOT_CACHE_WEIGHTING` | `--moe-hot-cache-weighting MODE` / `moe-hot-cache-weighting = MODE` | `flat`, `pressure`, `smooth`, `time`, `balanced` | `flat` | Waehlt den Ranking-Modus fuer initiale Hot-Cache-Auswahl und dynamisches Update. `flat` verteilt das Budget moeglichst gleichmaessig ueber die beobachteten Layer, indem Experten zuerst je Layer nach Hits sortiert und dann gleiche Raenge ueber die Layer interleaved werden. `pressure` stellt den vorherigen druckgewichteten Default wieder her. |
| `LLAMA_ARG_MOE_LAYER_PERF_OUT` | `--moe-layer-perf-out FNAME` / `moe-layer-perf-out = FNAME` | Pfad zu JSON-Datei | unset | Server-Helfer fuer den ersten Profiling-Lauf. Aktiviert Perf/Expert-Counts und schreibt die aktuelle `/moe-layer-perf`-JSON nach abgeschlossenen Requests und beim Shutdown in diese Datei. |

## Env-only Laufzeit- und Entwicklungshebel

| Env-Variable | Passendes Argument / INI-Key | Optionen | Default | Wirkung |
|---|---|---|---|---|
| `LLAMA_MOE_LAYER_PERF` | Kein direktes CLI-Argument. Laufzeitwechsel via `POST /moe-layer-perf` mit `{"mode":"full"}`, `{"mode":"update"}` oder `{"mode":"off"}`. `--no-perf` startet initial in `off`. | `full`, `update`, `off`; Aliase: `1`/`on`/`true` fuer `full`, `0`/`false` fuer `off` | `full`, ausser bei `--no-perf`: `off` | Steuert, welche MoE-Perf-Zaehler aktiv sind. `full` sammelt alle Timing- und Expertendaten, `update` nur die fuer dynamisches Update/Hitrate noetigen Daten, `off` schaltet den Pfad aus. |
| `LLAMA_MOE_HOT_CACHE_PARALLEL` | Kein Argument | unset/leer/`1`/`on`/`true`/`auto` = Auto, `0`/`off`/`false` = aus, `force` = erzwingen | Auto | Aktiviert die Hot/Cold-Fork-Join-Region im Scheduler. Auto faellt bei unguenstigen Regionen seriell zurueck, `force` macht daraus einen Fehler und ist fuer Debugging gedacht. |
| `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS` | Kein Argument | Ganzzahl `>= 0` | `2` | Mindestanzahl Hot+Cold-Slots, ab der Auto-Parallelisierung gestartet wird. `0` bedeutet: Parallelregion immer versuchen. |
| `LLAMA_MOE_HOT_CACHE_GEMMA4_LAYER_CURVE` | Kein Argument | Float `0.0` bis `1.0` | `0.5` | Gemma4-spezifische Layer-Druck-Gewichtung fuer initiale Auswahl und dynamisches Update. Entspricht konzeptionell der Qwen-Kurve, hat aber noch kein CLI-/INI-Argument. |
| `LLAMA_MOE_HOT_CACHE_BRANCH_REDUCE_MERGE` | Kein Argument | Boolean | on | Aktiviert den Branch-Reduce-Merge-Pfad. Effektiv aktuell fuer Gemma4: Hot- und Cold-Lane reduzieren ihre Slot-Ausgaben vor dem finalen Join. Qwen35Moe setzt diesen Profil-Schalter explizit aus. |
| `LLAMA_MOE_HOT_CACHE_MERGE_SUM_ROWS` | Kein Argument | Boolean | on | Nutzt einen optimierten Summenpfad, um mehrere MoE-Slot-Ausgaben in das finale Token-Ergebnis zu reduzieren. |
| `LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING` | Kein Argument | Boolean | on | Verschiebt die Hot/Cold-Routing- und Worklist-Erzeugung im Decode auf einen CPU-Custom-Op-Pfad. |
| `LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE` | Kein Argument | Boolean | on, aber architekturabhaengig | Qwen-Decode-Optimierung: merged Single-Token-Decode direkt in das finale Ergebnis, statt erst groessere Slot-Zwischenformen zu erzeugen. Gemma4 deaktiviert diesen Profilpfad aktuell explizit. |
| `LLAMA_MOE_HOT_CACHE_DECODE_STRIDED_SUM_ROWS` | Kein Argument | Boolean | on | Optimierter Summenpfad fuer Decode-Merge, wenn mehrere Slot-Zeilen reduziert werden muessen. |
| `LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING` | Kein Argument | Boolean | on | Fuegt Dummy-Hot-Arbeit hinzu, damit Graphform, Worklist und Backend-Scheduler bei leeren/kleinen Hot-Lanes stabil bleiben. |
| `LLAMA_MOE_HOT_CACHE_SHARED_INPUT_ROW` | Kein Argument | Boolean | on | Erlaubt im Decode, gleiche Input-Zeilen fuer Cold-Arbeit zu teilen, wenn alle Cold-Experten denselben Token-Input verwenden. |
| `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM` | Kein Argument | Boolean | on, aber architekturabhaengig | Cold-Decode-Optimierung: behandelt valide Cold-Slots als kompakten Prefix und reduziert nur diesen Bereich. Fuer Qwen aktivierbar; Gemma4 deaktiviert diesen Profilpfad aktuell explizit. |
| `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_WEIGHTED_SUM` | Kein Argument | Boolean | on, aber nur zusammen mit `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM` | Faltet die Expert-Gewichte direkt in die Cold-Prefix-Summe ein und spart separate Gewichtungs-/Merge-Arbeit. |
| `LLAMA_MOE_HOT_CACHE_DECODE_REPEAT_HOT_INPUT` | Kein Argument | Boolean | on | Wiederholt den aktuellen Decode-Token fuer Hot-Arbeit direkt, statt ihn ueber einen separaten Gather-Pfad zu holen. |
| `LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT` | Kein Argument | Boolean | on | Nutzt zusammen mit `LLAMA_MOE_HOT_CACHE_SHARED_INPUT_ROW` die erste Input-Zeile als gemeinsame Cold-Eingabe und reduziert Gather-/Input-Overhead. |

## Veraltete oder verworfene Schalter

Diese Namen tauchen in alten Notizen oder verworfenen Experimenten auf, haben im aktuellen Branch aber keine aktive Codewirkung:

| Env-Variable | Ersatz / Status | Grund |
|---|---|---|
| `LLAMA_MOE_HOT_CACHE_JSON` | Ersetzt durch `--moe-hot-cache` bzw. `LLAMA_ARG_MOE_HOT_CACHE` | Alter lokaler Starter-Name fuer die Hot-Cache-JSON. |
| `LLAMA_MOE_HOT_CACHE_MAX_MIB` | Ersetzt durch `--moe-hot-cache-max-mib` bzw. `LLAMA_ARG_MOE_HOT_CACHE_MAX_MIB` | Alter lokaler Starter-Name fuer das Cache-Budget. |
| `LLAMA_MOE_HOT_CACHE_QWEN_GPU_COLD_MERGE` | Verworfen | Experiment verschob Merge-Arbeit auf GPU, erhoehe aber Cold-Lane-/Sync-Druck und war langsamer. |
| `LLAMA_MOE_HOT_CACHE_QWEN_COLD_PREFIX_TASKS` | Verworfen | Experiment teilte die CPU-Prefix-Summe in mehrere Tasks. 2 und 4 Tasks waren langsamer als der Single-Task-Default. |
