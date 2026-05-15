# MoE Hot-Cache Parallelisierung: Aenderungen, Gruende und Wirkung

Stand dieser Notiz: Analyse von `0641adb9e08da0a675058bc39a8c928a7f8d6ad0..HEAD`
auf Branch `cached-experts-v2` mit `HEAD = 1618089c0` sowie der zu diesem
Zeitpunkt noch uncommitteten Aenderung in `src/models/qwen35moe.cpp`.

## Ausgangspunkt und Ziel

Der Startpunkt war ein llama.cpp-Zustand vor der Hot-Cache-Arbeit. Das Ziel
war, Qwen3.5/Qwen3.6 MoE Decode auf einer GPU plus CPU schneller zu machen,
obwohl nicht alle Experten gleichzeitig sinnvoll auf die GPU passen.

Die Grundidee:

- haeufig genutzte Experten pro Layer als Hot-Cache auf die GPU kopieren,
- seltenere Experten auf der CPU belassen,
- pro Decode-Schritt die MoE-Auswahl in Hot- und Cold-Arbeit aufteilen,
- GPU-Hot-Lane und CPU-Cold-Lane parallel ausfuehren,
- danach beide Ergebnisse wieder zusammenfuehren.

Die Messlatte aus den Laeufen war grob:

- ohne Hot/Cold-Parallelisierung bei einem einfachen "Hallo": ca. `19.7 tk/s`,
- erste Hot/Cold-Variante: einfache Prompts ca. `13.96 tk/s`, Programmierprompt
  ca. `16.4-17 tk/s`,
- aktueller Stand mit `performance56.json`: ca. `28.09 tk/s`.

Die Werte sind nicht alle strikte A/B-Messungen, weil Prompt, Lauflaenge,
Hot-Cache-JSON, Perf-Zaehler und einzelne Toggles ueber die Iterationen
variiert haben. Die Richtung ist trotzdem eindeutig: aus der ersten
funktionierenden Parallelisierung wurde ein deutlich schnellerer Decode-Pfad.

## Betroffene Bereiche

Seit dem Start-Commit wurden im relevanten Bereich geaendert:

- `src/llama-moe-hot-cache.{h,cpp}`
- `src/models/qwen35moe.cpp`
- `src/llama-context.cpp`
- `src/llama-graph.{h,cpp}`
- `src/llama-model.{h,cpp}`
- `src/models/models.h`
- `ggml/include/ggml-backend.h`
- `ggml/include/ggml.h`
- `ggml/src/ggml-backend.cpp`
- `ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml/src/ggml-cpu/ops.cpp`
- `ggml/src/ggml-cuda/ggml-cuda.cu`
- `ggml/src/ggml-cuda/reduce_rows.cuh`
- `ggml/src/ggml-cuda/sumrows.cu`
- `tests/test-moe-hot-cache.cpp`

Zusaetzlich wurden lokal Hilfsdateien fuer die Tests und Serverstarts genutzt,
unter anderem `model_config.ini`, `start-server` und `start-server-performance14`.
Diese Dateien waren beim Erstellen dieser Notiz noch untracked und damit nicht
Teil von `HEAD`.

## Commit-Timeline

Die wesentlichen Commits seit `0641adb9e08da0a675058bc39a8c928a7f8d6ad0`:

- `5ad16adeb`: erster Inferenz-Test mit Hot-Cache-Integration. Korrektheit war
  wichtiger als Speed; die Geschwindigkeit fiel erwartungsgemaess.
- `c26ec8556`: Runnable PoC. Der Hot-Cache-Pfad wurde lauffaehig und bekam
  erste Tests.
- `61aaa4c75`: mehr MoE-Performance-Metriken, damit sichtbar wird, wo Zeit
  verloren geht.
- `338c5ee05`: erster Parallelisierungsversuch, noch kaputt.
- `2a795fc3a`: Revert des kaputten Parallelisierungsversuchs.
- `980ad5292`: Entfernen einzelner Meta-Dokumente im experimentellen Fork.
  Das hatte keinen Runtime-Effekt.
- `dac2548ac`: erste funktionierende Hot/Cold-Parallelisierung. CPU und GPU
  wurden getrennt gerechnet und parallel getimed.
- `39a942ab1`: Scheduler-Optimierung.
- `50284e9b8`: weitere Speed-Arbeit am Parallelpfad.
- `220b44da3`: neue Methode, um Hot-Experten aus den Perf-Daten zu bestimmen.
- `5759b13aa`: Gewichtung der Auswahl angepasst.
- `03b23a2f6`: Routing verbessert.
- `9987d8266`: Perf-Pfad auf die fuer Optimierung relevanten Daten reduziert.
- `19c309c84`: Perf-Zaehler per `--no-perf` sauber abschaltbar gemacht.
- `d7305c2d0`: weitere Performance-Arbeit, unter anderem am Merge-/Sum-Pfad.
- `a47d68214`: Cold-Pfad schneller gemacht.
- `1618089c0`: aktueller committeter Stand mit ca. `28 tk/s`.

Danach lag noch eine uncommittete Aenderung in `src/models/qwen35moe.cpp` vor:
`LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT`. Diese reduzierte in
`performance56.json` den Worklist-/Cold-Gather-Anteil, verschob aber etwas Zeit
in Merge, Cold-Lane und Join-Wait.

## Hot-Cache-Datenmodell und Laden der Experten

### Was wurde geaendert

`src/llama-moe-hot-cache.{h,cpp}` fuehrt ein eigenes Datenmodell fuer den
MoE-Hot-Cache ein:

- `llama_moe_hot_cache_entry`
- `llama_moe_hot_cache_expert_size`
- `llama_moe_hot_cache_plan`
- `llama_moe_hot_cache_layer`
- `llama_moe_hot_cache`

Die Initialisierung liest eine Perf-JSON-Datei, bewertet Experten pro Layer,
plant die Auswahl innerhalb eines MiB-Budgets und kopiert die ausgewaehlten
Expertentensoren in GPU-nahe Hot-Cache-Tensoren. Unterstuetzt werden dabei die
Qwen-MoE-FFN-Tensoren fuer Gate, Up, Down sowie der fused Gate-Up-Fall.

Auf Modellebene wurden Parameter fuer Pfad und Budget ergaenzt:

- `--moe-hot-cache <json>`
- `--moe-hot-cache-max-mib <mib>`

### Warum

Alle Experten auf der GPU zu halten ist fuer das Zielmodell nicht der sinnvolle
Pfad. Gleichzeitig ist der Decode-Fall stark wiederholend: viele Layer und
Prompts nutzen bestimmte Experten deutlich haeufiger. Der Cache soll deshalb
GPU-Speicher fuer die Experten ausgeben, die den meisten Cold-Pfad vermeiden.

### Wirkung

Der Hot-Cache ist die Grundlage fuer die ganze Beschleunigung. In den spaeteren
Laeufen lag die Hot-Slot-Quote stabil um ca. `68-69%`. Beispiel
`performance56.json`:

- `n_expert = 256`
- `n_expert_used = 8`
- `hot_slot_ratio = 0.683649`

Damit bleiben ca. 31-32% der MoE-Slots im Cold-Pfad. Das passt zu der Annahme,
dass die erreichbare GPU-Hitrate vermutlich bei ungefaehr 70% gedeckelt ist.

## Qwen3.5-MoE-Graph: Hot/Cold-Split

### Was wurde geaendert

`src/models/qwen35moe.cpp` baut fuer aktive Hot-Cache-Layer nicht mehr den
normalen MoE-FFN-Pfad, sondern einen spezialisierten Hot/Cold-Pfad:

1. Router-Logits berechnen.
2. Worklist erzeugen.
3. Worklist in Hot-IDs, Cold-IDs, Gewichte, Token-/Slot-IDs und Counts
   aufteilen.
4. Hot-Branch mit gecachten Experten rechnen.
5. Cold-Branch mit normalen Layer-Experten rechnen.
6. Ergebnisse mergen.

Die Worklist-Felder sind:

- `HOT_ID`
- `HOT_SRC_SLOT`
- `HOT_TOKEN_ID`
- `HOT_WEIGHT`
- `COLD_ID`
- `COLD_SRC_SLOT`
- `COLD_TOKEN_ID`
- `COLD_WEIGHT`
- `HOT_EXPERT_ID`
- `HOT_COUNT`
- `COLD_COUNT`

Der gemeinsame FFN-Helfer wurde so erweitert, dass er externe Expert-IDs,
kompakte Gewichte und Branch-spezifische Backends akzeptiert. Dadurch kann der
Hot-Branch explizit CUDA und der Cold-Branch explizit CPU verwenden.

### Warum

Der normale MoE-Pfad rechnet eine gemeinsame Expertenauswahl. Fuer parallele
Ausfuehrung brauchen wir aber zwei unabhaengige Arbeitsmengen:

- Hot: die Experten, die im GPU-Cache liegen.
- Cold: alle uebrigen Experten.

Erst wenn diese beiden Mengen kompakt und eindeutig markiert sind, kann der
Scheduler sie als Fork/Join-Region behandeln.

### Wirkung

Der Graph-Split machte die Hot/Cold-Parallelisierung ueberhaupt moeglich. Er
war auch der Bereich, in dem die ersten Fehler auftraten:

- ein `ggml_view_1d` Bounds-Assert bei der ersten Inferenz,
- danach ein Scheduler-Fehler: `region split order is not hot-then-cold-then-join`.

Die Loesung war, die Region expliziter zu bauen, die Counts als Prefix-Views
voranzustellen und im Scheduler streng zu validieren, dass die Split-Reihenfolge
Hot -> Cold -> Join ist.

## CPU-Routing und Worklist-Erzeugung

### Was wurde geaendert

Fuer Decode mit `n_tokens == 1` wurde ein CPU-Custom-Op ergaenzt:

- `llama_qwen35moe_hot_cache_build_worklist_from_logits_op`

Dieser Op berechnet Top-K, Softmax/Normierung, Gewichtsskalierung und Hot/Cold
Packing direkt aus den Router-Logits. Fuer Prefill oder andere Faelle gibt es
weiterhin den Pfad ueber:

- `ggml_argsort_top_k`
- `ggml_get_rows`
- `ggml_soft_max`
- `llama_qwen35moe_hot_cache_build_worklist_op`

Der Decode-CPU-Routing-Pfad ist standardmaessig aktiv und per Env abschaltbar:

- `LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING=0`

### Warum

Bei Decode ist `n_tokens == 1`. Der normale Graph-Pfad erzeugt dafuer viel
kleine Arbeit: Argsort, GetRows, Softmax, Views und Casts. Diese Arbeit ist
relativ teuer im Verhaeltnis zur eigentlichen MoE-Matmul-Arbeit. Ausserdem
erzeugten die vielen Views frueh fragile Graph-Kanten.

### Wirkung

Routing und Worklist wurden stabiler und messbar kleiner. In den aktuellen
Metriken ist Routing zwar immer noch ein grosser Block, aber die spaeteren
Optimierungen konnten die Worklist-Zeit stark senken:

- `performance45.json`: `worklist_time_per_call_us = 144.66`
- `performance52.json`: `worklist_time_per_call_us = 113.361`
- `performance56.json`: `worklist_time_per_call_us = 65.3677`

## Scheduler: parallele Hot/Cold-Region

### Was wurde geaendert

In `ggml/include/ggml-backend.h` und `ggml/src/ggml-backend.cpp` wurde eine
experimentelle Scheduler-Annotation eingefuehrt:

- `ggml_backend_sched_moe_hot_cache_parallel_region(...)`
- `ggml_backend_sched_set_moe_hot_cache_parallel_perf_enabled(...)`
- `ggml_backend_sched_get_moe_hot_cache_parallel_perf(...)`

Der Scheduler erkennt damit eine Fork/Join-Region:

- Hot-Start bis Hot-End auf CUDA,
- Cold-Start bis Cold-End auf CPU,
- Join-Knoten danach.

Die Cold-Lane wird ueber einen CPU-Worker-Thread parallel zur Hot-Lane
ausgefuehrt. Der Thread selbst laeuft auf der CPU; GPU-Arbeit entsteht nur,
wenn die Hot-Lane CUDA-Kernel an das CUDA-Backend uebergibt.

Der Parallelmodus wird gesteuert ueber:

- `LLAMA_MOE_HOT_CACHE_PARALLEL=0`: aus
- `LLAMA_MOE_HOT_CACHE_PARALLEL=1`: auto
- `LLAMA_MOE_HOT_CACHE_PARALLEL=force`: erzwingen
- `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS`: Mindestanzahl Slots

Wichtig: Im Code ist Parallel ohne Env aus. Das lokale `start-server` setzt es
standardmaessig auf `1` und `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS=0`.

### Warum

Ohne Scheduler-Erweiterung rechnet llama.cpp die Backend-Splits seriell. Dann
spart der GPU-Hot-Cache zwar Cold-Arbeit, aber CPU- und GPU-Arbeit ueberlappen
nicht sauber. Das Ziel war, die ungenutzte CPU/GPU-Zeit waehrend Decode
auszunutzen.

### Wirkung

Dies war der erste grosse Hebel. Nach dem funktionierenden Parallelpfad wurde
die 20-tk/s-Marke erreicht und spaeter deutlich ueberschritten.

In `performance56.json` sieht man den aktuellen Zustand:

- `parallel_region_wall_time_per_call_us = 422.892`
- `parallel_hot_lane_wall_time_per_call_us = 123.504`
- `parallel_cold_lane_wall_time_per_call_us = 382.102`
- `parallel_join_wait_time_per_call_us = 332.197`
- `parallel_fallbacks = 0`

Das bedeutet: Die Parallelregion laeuft stabil ohne Fallbacks, aber die
Cold-Lane ist weiterhin der laengere Ast. Der Join wartet daher oft auf den
Cold-Pfad.

## Backend- und GGML-Op-Erweiterungen

### `mul_mat_id` Flags

`src/llama-graph.h` bekam neue Flags:

- `LLM_MUL_MAT_ID_FLAG_ALLOW_DUPLICATE_IDS`
- `LLM_MUL_MAT_ID_FLAG_ALLOW_NEGATIVE_IDS`
- `LLM_MUL_MAT_ID_FLAG_SKIP_NEGATIVE_ID_OUTPUT_ZERO`
- `LLM_MUL_MAT_ID_FLAG_SHARED_INPUT_ROW`

Diese Flags wurden in `src/llama-graph.cpp`, CPU-Backend und CUDA-Backend
angebunden.

### Warum

Der kompakte Hot/Cold-Pfad hat andere Anforderungen als der Standard-MoE-Pfad:

- Padding-Slots koennen negative IDs enthalten.
- Decode nutzt denselben Input-Token fuer viele Expert-Slots.
- Unbenutzte Padding-Ausgaben muessen nicht zwingend genullt werden, wenn sie
  spaeter garantiert ignoriert werden.
- Duplicate IDs sind in der kompakten Darstellung erlaubt.

### Wirkung

Die Flags erlauben kleinere und guenstigere Branches:

- weniger unnoetiges Zeroing,
- weniger Input-Kopien,
- weniger Arbeit fuer ungueltige Padding-Slots,
- Grundlage fuer `SHARED_INPUT_ROW` und spaetere Cold-Pfad-Optimierungen.

### `get_rows` First-Row-Only

In `ggml/include/ggml.h` wurde `GGML_GET_ROWS_FLAG_FIRST_ROW_ONLY` ergaenzt und
im CPU-GetRows-Pfad genutzt.

Warum: Bei Decode ist der Input nur eine Zeile. Der Cold-Pfad muss fuer mehrere
Expert-Slots nicht immer wieder dieselbe Aktivierung voll kopieren.

Wirkung: Der Cold-Gather-Anteil sank spaeter deutlich. In `performance56.json`
liegt `cold_gather_scatter_time_per_call_us` nur noch bei `2.56514`, waehrend er
in `performance55.json` noch `10.2372` war.

### `sum_rows` und strided Inputs

CPU- und CUDA-`sum_rows` wurden erweitert, unter anderem:

- CPU kann kompatible Faelle parallelisieren,
- CPU kommt mit strided Input besser zurecht,
- CUDA bekam einen strided F32-Reduce-Pfad in `reduce_rows.cuh` /
  `sumrows.cu`.

Warum: Der Merge-Pfad erzeugt kompakte bzw. strided Darstellungen. Ein
erzwungenes `cont` oder Summieren ueber volle Kapazitaet kostet viel.

Wirkung: Diese Arbeit war Grundlage fuer Direct-Merge, Prefix-Sum und spaetere
Merge-Optimierungen. Merge ist trotzdem weiterhin der groesste Block.

## Decode-Merge, Prefix-Sum und Graph-Pruning

### Direct Decode Merge

Fuer Decode kann der Hot/Cold-Pfad direkt ein `[n_embd, 1]` Ergebnis erzeugen,
statt erst ein volles Slot-Tensorlayout aufzubauen und danach zu summieren.

Schalter:

- `LLAMA_MOE_HOT_CACHE_DECODE_DIRECT_MERGE`

Default: aktiv.

Warum: Bei `n_tokens == 1` ist das volle Slot-Layout nur Zwischenarbeit.

Wirkung: Weniger Views, weniger `set_rows`, weniger Sum-Arbeit.

### Cold Prefix Sum

Der Cold-Pfad sortiert gueltige Cold-Slots als Prefix. Statt ueber die volle
Kapazitaet inklusive Padding zu summieren, wird nur der gueltige Prefix
verarbeitet.

Schalter:

- `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_SUM`

Default: aktiv.

Warum: Padding-Slots tragen nichts bei, kosten aber im Merge und in
CPU/CUDA-Backendpfaden trotzdem Zeit.

### Weighted Cold Prefix Sum

Der Cold-Pfad kann Gewichtung und Prefix-Summe in einem Custom-Op verbinden:

- `llama_qwen35moe_hot_cache_sum_weighted_prefix_rows_op`

Dabei wird im Cold-FFN `apply_weights=false` verwendet; die Gewichtung erfolgt
beim Prefix-Merge.

Schalter:

- `LLAMA_MOE_HOT_CACHE_COLD_PREFIX_WEIGHTED_SUM`

Default: aktiv.

Wirkung im direkten Vergleich:

- `performance52.json -> performance53.json`
- `total_moe_time_per_call_us`: `900.611 -> 884.981` (`-15.63 us`)
- `merge_time_per_call_us`: `475.775 -> 459.535` (`-16.24 us`)
- `parallel_cold_lane_wall_time_per_call_us`: `383.962 -> 369.964` (`-14.00 us`)
- `parallel_join_wait_time_per_call_us`: `327.476 -> 310.884` (`-16.59 us`)

### Graph-Pruning

Spaeter wurden ungenutzte Worklist-Views im Decode-Pfad gar nicht mehr gebaut:

- `hot_src_slots` nur, wenn kein Direct-Merge aktiv ist,
- `cold_src_slots` nur, wenn kein Direct-Merge aktiv ist,
- `cold_weights` nicht, wenn Weighted-Prefix-Sum aktiv ist,
- `hot_expert_ids` nur, wenn Expert-Counts wirklich eingeschaltet sind.

Wirkung:

- `performance53.json -> performance54.json`
- `total_moe_time_per_call_us`: `884.981 -> 842.767` (`-42.214 us`)
- `worklist_time_per_call_us`: `113.256 -> 87.951` (`-25.305 us`)
- `hot_gather_scatter_time_per_call_us`: `22.807 -> 12.9825` (`-9.8245 us`)
- `cold_gather_scatter_time_per_call_us`: `19.8564 -> 10.2588` (`-9.5976 us`)

Das war einer der groessten spaeten Einzelhebel.

### Repeat Hot Input

Fuer Decode kann der Hot-Input per `ggml_repeat_4d(cur, capacity)` erzeugt
werden, statt `hot_token_ids` plus `get_rows` zu benutzen.

Schalter:

- `LLAMA_MOE_HOT_CACHE_DECODE_REPEAT_HOT_INPUT`

Default: aktiv.

Wirkung:

- `performance54.json -> performance55.json`
- `total_moe_time_per_call_us`: `842.767 -> 820.799` (`-21.968 us`)
- `worklist_time_per_call_us`: `87.951 -> 76.8338` (`-11.1172 us`)
- `hot_gather_scatter_time_per_call_us`: `12.9825 -> 4.95083` (`-8.03167 us`)

### Cold First-Row Input

Die zuletzt uncommittete Optimierung baut fuer den Cold-Pfad bei Decode eine
Input-Matrix aus der ersten Zeile, statt per GetRows wiederholt dieselbe Zeile
zu kopieren.

Schalter:

- `LLAMA_MOE_HOT_CACHE_COLD_FIRST_ROW_INPUT`

Default im aktuellen Arbeitsbaum: aktiv.

Wirkung:

- `performance55.json -> performance56.json`
- `total_moe_time_per_call_us`: `820.799 -> 816.307` (`-4.492 us`)
- `worklist_time_per_call_us`: `76.8338 -> 65.3677` (`-11.4661 us`)
- `cold_gather_scatter_time_per_call_us`: `10.2372 -> 2.56514` (`-7.67206 us`)

Aber:

- `merge_time_per_call_us`: `460.041 -> 471.037` (`+10.996 us`)
- `parallel_cold_lane_wall_time_per_call_us`: `369.626 -> 382.102` (`+12.476 us`)
- `parallel_join_wait_time_per_call_us`: `313.532 -> 332.197` (`+18.665 us`)

Interpretation: netto leicht positiv im gemessenen Lauf, aber nicht eindeutig
dominant. Dieser Schalter sollte weiter A/B getestet werden.

## Perf-Daten und `--no-perf`

### Was wurde geaendert

`src/llama-context.cpp` bekam eine spezialisierte MoE-Perf-Erfassung. Die alte
Heatmap-Sicht wurde fuer Optimierung zurueckgebaut und durch ein schlankeres
Schema ersetzt:

- `llama.cpp.moe_layer_opt_perf.v1`

Die JSON enthaelt heute vor allem:

- Summary ueber alle Layer,
- Hot-/Cold-Slot-Verhaeltnis,
- Zeiten fuer Routing, Worklist, Merge,
- Hot-/Cold-Branch-Zeiten,
- Hot-/Cold-Gather/Scatter,
- Parallelregion, Hot-Lane, Cold-Lane, Join-Wait,
- Fallback-Zaehler und Fallback-Gruende.

Expert-Counts sind optional:

- `LLAMA_MOE_LAYER_PERF_EXPERT_COUNTS=1`

Die Perf-Erfassung ist nur aktiv, wenn sie per Env aktiviert ist und der Kontext
nicht `no_perf` gesetzt hat. `--no-perf` deaktiviert dadurch auch die
MoE-Layer-Perf und die Scheduler-Parallel-Perf.

### Warum

Die Visualisierung der Layer/Experten-Auslastung war fuer den ersten Cacheplan
nuetzlich, aber fuer Speed-Optimierung brauchten wir andere Daten:

- Wo ist die Zeit?
- Ist die Cold-Lane laenger als die Hot-Lane?
- Wartet der Join?
- Gibt es Scheduler-Fallbacks?
- Ist Merge oder Routing groesser als die eigentlichen Matmuls?

### Wirkung

Die Spaetphase der Optimierung war nur durch diese Metriken zielgerichtet
moeglich. Beispiel `performance56.json`:

- `total_moe_time_per_call_us = 816.307`
- `routing_time_per_call_us = 264.614`
- `worklist_time_per_call_us = 65.3677`
- `merge_time_per_call_us = 471.037`
- `parallel_region_wall_time_per_call_us = 422.892`
- `parallel_hot_lane_wall_time_per_call_us = 123.504`
- `parallel_cold_lane_wall_time_per_call_us = 382.102`
- `parallel_join_wait_time_per_call_us = 332.197`
- `parallel_fallbacks = 0`

Wichtig: Diese Kategorien sind nicht alle additiv. Parallelregion, Branch- und
Merge-Zeiten ueberlappen sich konzeptionell teilweise. Als Diagnose zeigen sie
aber klar: Merge, Cold-Lane und Join-Wait sind aktuell die dominanten Themen.

## Hot-Expert-Auswahl aus Perf-JSON

### Was wurde geaendert

Der Parser in `llama_moe_hot_cache_parse_perf_json` akzeptiert jetzt:

- `llama.cpp.moe_layer_perf.v1`
- `llama.cpp.moe_layer_opt_perf.v1`

Er kann sowohl alte rohe `experts`-Counts als auch neue Branch-Counts lesen:

- `hot_experts`
- `cold_experts`

Die Bewertung beruecksichtigt ausserdem Layer-Gewichtung anhand von
Cold-Wartekosten:

- `parallel_join_wait_time_per_call_us`
- `cold_slots_per_call`
- alternativ Differenz aus Cold-Lane und Hot-Lane

Die Gewichtung wird gedaempft und begrenzt, damit der Cache nicht bei jedem Lauf
stark churnt. Bereits hot gewesene Experten bekommen einen kleinen Sticky-Bonus.

### Warum

Eine reine Hit-Heatmap reicht nicht. Ein Experte in einem Layer mit wenig
Cold-Wait ist weniger wert als ein Experte in einem Layer, in dem die CPU-Lane
den Join ausbremst. Der Cache soll GPU-Speicher dort einsetzen, wo er
Join-Wartezeit reduziert.

### Wirkung

Die Hot-Slot-Quote blieb nicht maximal, aber besser auf die teuren Layer
ausgerichtet. Das erklaert, warum spaetere Laeufe trotz Hot-Slot-Ratio um 68-69%
schneller wurden als fruehere Laeufe mit teilweise hoeherer Ratio. Beispiel:

- `performance45.json`: `hot_slot_ratio = 0.762117`, aber
  `total_moe_time_per_call_us = 944.125`
- `performance56.json`: `hot_slot_ratio = 0.683649`, aber
  `total_moe_time_per_call_us = 816.307`

Mehr Hot-Hits allein sind also nicht automatisch besser; entscheidend ist, ob
die Cold-Tail-Zeit sinkt.

## Start-Skripte und lokale Konfiguration

### `model_config.ini`

Die lokale Config nutzt fuer Qwen3.6-35B-A3B unter anderem:

- `device = CUDA0`
- `ngl = 999`
- `cpu-moe = true`
- `ctx-size = 32000`
- `override-kv = qwen35moe.expert_used_count=int:8`

Der wichtige Punkt ist `n_expert_used = 8`. Die gesamte Optimierung und die
gemessenen Perf-Daten beziehen sich auf diesen Top-K-Wert.

### `start-server`

Das lokale `start-server` setzt Defaults, damit nicht jedes Mal Env-Variablen
manuell exportiert werden muessen:

- `LLAMA_MOE_HOT_CACHE_PARALLEL=1`
- `LLAMA_MOE_HOT_CACHE_PARALLEL_MIN_SLOTS=0`
- `LLAMA_MOE_HOT_CACHE_JSON=/home/adrian/models/heatmap-data.json`
- `LLAMA_MOE_HOT_CACHE_MAX_MIB=8000`

Danach startet es `llama-server` mit:

- `--models-preset /home/adrian/llama.exp/model_config.ini`
- `--moe-hot-cache-max-mib "$LLAMA_MOE_HOT_CACHE_MAX_MIB"`
- `--moe-hot-cache "$LLAMA_MOE_HOT_CACHE_JSON"`

### `start-server-performance14`

Eine zweite lokale Variante startet mit:

- `LLAMA_MOE_HOT_CACHE_JSON=/home/adrian/llama.exp/performance14.json`

Das war nuetzlich, um reproduzierbar gegen ein bestimmtes Hot-Cache-JSON zu
testen.

## Tests

`tests/test-moe-hot-cache.cpp` wurde neu aufgebaut bzw. erweitert. Abgedeckt
sind:

- Parse und Sortierung alter Perf-JSON-Daten,
- Parse von Hot-/Cold-Branch-Counts plus Layer-Gewichtung,
- Budget-Auswahl fuer Experten,
- Fehler bei falschem Schema,
- gemischte Hot/Cold-Worklist,
- All-Hot- und All-Cold-Faelle,
- Worklist-Erzeugung direkt aus Router-Logits inklusive Gewichtung.

Warum: Der Hot/Cold-Pfad ist sehr empfindlich gegen kleine Layoutfehler. Wenn
ein Count, Slot oder Padding-Wert falsch ist, entstehen entweder falsche
Ausgaben oder Scheduler-Regionen, die nicht parallelisiert werden koennen.

Wirkung: Die Tests sichern die Logik ab, die unabhaengig von einem grossen
Qwen-Modell sinnvoll testbar ist. Die wirklich backendspezifischen Teile
brauchen weiterhin echte Server-/CLI-Laeufe.

## Performance-Verlauf

Grobe Entwicklung aus den user-reported Laeufen:

| Phase | Ergebnis | Bedeutung |
| --- | ---: | --- |
| Ohne Parallelisierung, einfaches Hallo | ca. `19.7 tk/s` | Vergleichswert fuer normalen Pfad |
| Erste Parallel-Variante, einfaches Hallo | ca. `13.96 tk/s` | Korrekt, aber zu viel Overhead |
| Erste Programmierlaeufe | ca. `16.4-17 tk/s` | Hot-Cache passt besser zum Programmierprompt |
| `performance9` / `performance14` | ca. `20 tk/s` | erste grosse Parallelgewinne |
| `performance41-47` | ca. `23.7-24.75 tk/s` | Scheduler-/Routing-/Cache-Auswahl verbessert |
| `performance50-52` | ca. `25.45-25.49 tk/s` | Cold-/Merge-Arbeit weiter reduziert |
| `performance53` | knapp `26 tk/s` | Weighted Prefix Sum und Cold-Pfad-Arbeit |
| `performance54` | ca. `27.1 tk/s` | Graph-Pruning und weniger ungenutzte Views |
| `performance55` | ca. `27.76 tk/s` | Repeat-Hot-Input |
| `performance56` | ca. `28.09 tk/s` | aktueller Stand inklusive Cold-First-Row-Test |

## Aktuelle Engpaesse laut `performance56.json`

Die wichtigsten Summary-Werte:

| Metrik | Wert |
| --- | ---: |
| `total_moe_time_per_call_us` | `816.307` |
| `routing_time_per_call_us` | `264.614` |
| `worklist_time_per_call_us` | `65.3677` |
| `merge_time_per_call_us` | `471.037` |
| `hot_gather_scatter_time_per_call_us` | `5.00415` |
| `cold_gather_scatter_time_per_call_us` | `2.56514` |
| `parallel_region_wall_time_per_call_us` | `422.892` |
| `parallel_hot_lane_wall_time_per_call_us` | `123.504` |
| `parallel_cold_lane_wall_time_per_call_us` | `382.102` |
| `parallel_join_wait_time_per_call_us` | `332.197` |
| `parallel_fallbacks` | `0` |

Interpretation:

- Merge ist mit ca. 58% von `total_moe_time_per_call_us` der groesste
  ausgewiesene Einzelblock.
- Routing ist mit ca. 32% weiterhin gross.
- Worklist ist deutlich kleiner geworden, aber noch sichtbar.
- Gather/Scatter ist nach den letzten Optimierungen klein.
- Die Parallelregion funktioniert ohne Fallbacks.
- Die Cold-Lane ist viel laenger als die Hot-Lane; der Join wartet daher weiter
  stark auf CPU-Cold-Arbeit.

## Was die Arbeit insgesamt gebracht hat

Funktional:

- Hot-Cache kann aus Perf-JSON geplant und geladen werden.
- Qwen3.5/Qwen3.6-MoE kann Hot- und Cold-Experten getrennt rechnen.
- CPU-Cold und CUDA-Hot laufen in einer Scheduler-Region parallel.
- Erste-Inferenz-Crashes und Split-Order-Fehler wurden beseitigt.
- `--no-perf` kann die Performance-Zaehler fuer reine Speed-Laeufe abschalten.

Performance:

- Die erste funktionierende Variante war noch langsamer als der normale Pfad.
- Durch Scheduler-, Routing-, Worklist-, Merge- und Cold-Pfad-Arbeit stieg der
  Programmierprompt von ca. `16-17 tk/s` auf ca. `28.09 tk/s`.
- Gegenueber dem fruehen funktionierenden Parallelpfad ist das grob `+65%`.
- Gegenueber dem einfachen `19.7 tk/s` Vergleichswert ist der aktuelle
  Programmierlauf grob `+43%`, wobei das kein sauberer A/B-Vergleich ist.

Diagnostisch:

- Die Perf-Daten zeigen jetzt, dass nicht mehr die Heatmap selbst das Problem
  ist, sondern Merge, Cold-Lane, Join-Wait und Routing.
- Die Hot-Hitrate ist nahe am erwarteten Limit; weitere grosse Gewinne muessen
  wahrscheinlich aus weniger Overhead und einem kuerzeren Cold-Tail kommen.

## Naechste technische Hebel

Die sinnvollsten naechsten Ansatzpunkte sind aus heutiger Sicht:

1. Merge weiter reduzieren
   - `merge_time_per_call_us` ist der groesste Block.
   - Moegliche Richtung: Hot- und Cold-Ergebnis frueher in ein gemeinsames
     `[n_embd, 1]` Ergebnis falten, weniger Add-/Sum-/View-Knoten.

2. Cold-Lane verkuerzen
   - Die Cold-Lane bestimmt den Join.
   - Moegliche Richtung: CPU-Cold-Matmul und Prefix-Sum enger zusammenlegen,
     weniger Zwischenlayouts, bessere aktive-ID-Liste.

3. Routing reduzieren
   - Routing liegt weiterhin bei ca. `265 us` pro Layer-Call.
   - Moegliche Richtung: Top-K fuer `n_tokens == 1` weiter spezialisieren,
     weniger generische Tensorpfade, vielleicht direkte Router-Auswertung in
     einem noch schmaleren Op.

4. Cold-First-Row A/B klaeren
   - Netto in `performance56.json` leicht positiv, aber mit hoeherem Merge und
     Join-Wait.
   - Sollte als Toggle behalten und mit mehreren vergleichbaren Laeufen
     geprueft werden.

5. Dynamischer Cache
   - Ein Update nach Inferenzlaeufen koennte die Auswahl verbessern, wird aber
     nur helfen, wenn dadurch Cold-Tail-Layer getroffen werden.
   - Da die Hot-Hitrate vermutlich nahe 70% gedeckelt ist, sollte dynamisches
     Update nicht auf maximale Hitrate, sondern auf weniger Join-Wait optimieren.

