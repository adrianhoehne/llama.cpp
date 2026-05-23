# MoE Hot-Cache Code-Referenz

Stand: 2026-05-22

Diese Datei beschreibt die von uns eingeführte MoE-Hot-Cache-Implementierung aus den eigenen Commits auf `cached-experts-v2`. Sie ist bewusst code-nah geschrieben: Welche Methode macht was, warum existiert sie, und wie hilft sie bei Performance oder Wartbarkeit.

Abgedeckte eigene Commits:

| Commit | Thema |
| --- | --- |
| `25c231874` | MoE-Hot-Cache Runtime, Graph, Scheduler-Erweiterung, Perf-JSON, Server-API, Tests |
| `ce729f52f` | MoE-Layer-Performance-UI und Runtime-Perf-Modus-Schalter |
| `5a6edbb43` | Workflow- und Tuning-Dokumentation |
| `2b4944906` | Experiment-Dokumentation zu MTP und Quadro-M1200-Warm-Lane |
| `1749fbc83` | Gemma4 Cold-Prefix-Merge Optimierung |
| `27de9be40` | Aktuelle Doku-Updates nach Rebase und Experimenten |

## Architektur

Der Hot-Cache ist absichtlich als Zusatzpfad gebaut:

```text
Original llama.cpp Modell
├── alle MoE-Expert-Tensoren bleiben im normalen Modell
│   └── mit --cpu-moe typischerweise im CPU/RAM-Pfad
└── unser Hot-Cache
    ├── kopiert ausgewählte Expert-Slices zusätzlich in VRAM
    ├── baut pro Layer hot_id_map, hot_mask und cold_mask
    ├── splittet die Inferenz je Token/Expert-Slot in Hot- und Cold-Worklists
    ├── rechnet Hot-Branch auf GPU und Cold-Branch auf CPU
    └── merged beide Ausgaben wieder in den normalen Layer-Output
```

Wichtig: Der Hot-Cache spart aktuell keinen RAM. Er ist eine zusätzliche VRAM-Kopie ausgewählter Expert-Slices. Das ist Absicht, weil die Cold-Lane und dynamische Updates weiterhin auf die originalen CPU-Tensoren zugreifen müssen.

## Laufzeitfluss

1. `--moe-layer-perf-out <file>` oder `/moe-layer-perf` erzeugt eine Perf-JSON mit Experten-Hits, Hot/Cold-Slots und Timings.
2. `--moe-hot-cache <file> --moe-hot-cache-max-mib <N|-1> --cpu-moe` liest diese JSON.
3. `llama_moe_hot_cache_init()` bewertet die Experten, berechnet das Budget, kopiert die besten Expert-Slices in einen eigenen GPU-Buffer und setzt Maps/Masks.
4. Qwen35MoE und Gemma4 rufen pro Layer nur dann den Hot-Graph auf, wenn `llama_moe_hot_cache_layer_active()` true ist.
5. Der Graph baut eine Worklist. Diese trennt aktive Top-K-Expert-Slots in Hot und Cold.
6. Der Scheduler kann Hot- und Cold-Splits parallel ausführen.
7. Nach einem Server-Request kann `--moe-hot-cache-update-rate` einen Teil der Hot-Cache-Slots gegen aktuell bessere Experten austauschen.

## Datenstrukturen

Datei: `src/llama-moe-hot-cache.h`

| Name | Was sie enthält | Warum hilfreich |
| --- | --- | --- |
| `llama_moe_hot_cache_entry` | `(layer, expert, hit_count)` nach Scoring. | Einheitliches Ranking-Format für Initialbefüllung und Tests. |
| `llama_moe_hot_cache_expert_observation` | Roh-, Hot- und Cold-Hit-Zähler pro Expert. | Kann alte `experts`-Listen und neue `hot_experts`/`cold_experts` gleichzeitig lesen. |
| `llama_moe_hot_cache_layer_observation` | Layer-ID, Expertenliste und Timing-Felder wie Join-Wait, Cold-Lane-Zeit und MoE-Zeit. | Grundlage für Weighting-Modi, die nicht nur Hits, sondern auch langsame Layer bevorzugen. |
| `llama_moe_hot_cache_expert_size` | Speicherbedarf eines Expert-Slices. | Erlaubt Budget-Auswahl nach MiB statt nach bloßer Anzahl. |
| `llama_moe_hot_cache_plan` | Gesamtranking, ausgewählte Experten, Budget und belegte Bytes. | Macht die Cache-Auswahl testbar und loggbar. |
| `llama_moe_hot_cache_update_stats` | Hit-Rate, Kandidaten, Austauschbudget und getauschte Layer. | Liefert die Server-Logs nach dynamischem Update. |
| `llama_moe_hot_cache_weighting_mode` | `flat`, `pressure`, `smooth_pressure`, `time`, `balanced`. | Trennt Auswahlstrategie vom Cache-Allocator. |
| `llama_moe_hot_cache_weighting_config` | Weighting-Modus plus `layer_curve`. | Ein einzelner Config-Typ für Qwen, Gemma und spätere Modelle. |
| `llama_moe_hot_cache_layer` | Hot-Cache-Tensoren, Maps, Masks, Host-Map, `n_hot`, `n_expert`. | Repräsentiert einen aktiven Hot-Cache-Layer. |
| `llama_moe_hot_cache_layer::active()` | Prüft `n_hot`, `hot_id_map`, `hot_mask`, `cold_mask`. | Gate, damit normale Modelle und nicht gecachte Layer unverändert bleiben. |
| `llama_moe_hot_cache_worklist_field` | Feldlayout des Worklist-Tensors. | Hält Hot/Cold-IDs, Token-IDs, Gewichte und Counts in einem kompakten `ggml_tensor`. |
| `llama_moe_hot_cache` | Vektor aller Layer-Caches plus ggml-Kontexte und Backend-Buffer. | Hält die VRAM-Buffer am Leben, solange das Modell lebt. |
| `llama_moe_hot_cache::active()` | Prüft, ob mindestens ein Layer aktiv ist. | Schneller globaler Feature-Check. |

## Cache-Aufbau und Auswahl

Dateien: `src/llama-moe-hot-cache.cpp`, `src/llama.cpp`, `src/llama-context.cpp`

| Methode | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `llama_moe_hot_cache_hot_dummy_padding()` | Liest `LLAMA_MOE_HOT_CACHE_HOT_DUMMY_PADDING`; default an. | Negative oder ungültige Hot-IDs können über einen Dummy-Expert abgefangen werden, was CUDA-Pfade stabiler macht. |
| `key(layer, expert)` | Packt Layer und Expert in einen `uint64_t`. | Schneller Schlüssel für Maps ohne eigene Pair-Hash-Funktion. |
| `tensor_expert_bytes(t)` | Berechnet `ggml_nbytes(t) / t->ne[2]`. | Liefert die Kosten eines einzelnen Expert-Slices für Budgetplanung. |
| `add_saturating(dst, value)` | Addiert ohne Overflow; bei Overflow wird saturiert. | Perf-Zähler und Hit-Zähler können bei langen Läufen nicht überlaufen. |
| `select_gpu_dev(model)` | Nimmt die erste GPU/IGPU aus `model.devices`, sonst erste Backend-GPU. | Der Cache landet auf dem gleichen relevanten GPU-Backend wie das Modell. |
| `new_tensor_like_experts(ctx, src, n_cache, name)` | Erzeugt einen 3D-Hot-Cache-Tensor mit gleicher Expert-Matrixform, aber nur `n_cache` Experten. | Packt ausgewählte Experten dicht in VRAM statt komplette Layer zu kopieren. |
| `new_tensor_like_scale(ctx, src, n_cache, name)` | Erzeugt optionalen Scale-Tensor für quantisierte/scalierte Expert-Tensoren. | Hält quantisierte Expert-Pfade numerisch kompatibel. |
| `zero_tensor(t)` | Füllt einen Hot-Cache-Tensor mit Nullen. | Der Dummy-Expert und ungenutzte Slots liefern deterministisch Null. |
| `copy_expert_slice(src, dst, src_expert, dst_expert)` | Kopiert ein Expert-Slice aus dem Originaltensor in den Hot-Cache. | Das ist der eigentliche VRAM-Cache-Fill für `up`, `gate`, `gate_up`, `down`. |
| `copy_scale_slice(src, dst, src_expert, dst_expert)` | Kopiert den passenden Scale-Eintrag. | Verhindert falsche Skalierung bei quantisierten Expert-Slices. |
| `set_tensor_i32_1d()` / `set_tensor_f32_1d()` | Schreibt einzelne Map/Mask-Werte direkt in Backend-Tensoren. | Dynamische Updates können Maps ändern, ohne den ganzen Cache neu zu bauen. |
| `current_hot_experts(layer)` | Rekonstruiert aus `hot_id_map_host`, welche originalen Experten gerade im Cache liegen. | Basis für Austauschentscheidungen beim dynamischen Update. |
| `find_or_add_expert_observation(layer, expert)` | Findet einen Expert-Zähler oder legt ihn an. | JSON-Parsing kann `experts`, `hot_experts` und `cold_experts` zusammenführen. |
| `observation_total_hits(layer, expert)` | Nutzt `raw` oder `hot+cold` je nach JSON-Schema. | Unterstützt Lernlauf ohne Hot-Cache und spätere Läufe mit Hot/Cold-Split. |
| `sort_hot_cache_entries(entries)` | Sortiert nach Score absteigend, dann Layer und Expert. | Deterministische Auswahl, wichtig für reproduzierbare Tests. |
| `score_observations_default()` | Scort nur nach Hit-Anzahl. | Einfacher Fallback und Basistest für Perf-JSONs. |
| `weighting_config_from_params(params)` | Baut die Weighting-Config aus CLI-Parametern und Env-Fallbacks. | Macht `--moe-hot-cache-weighting` und `--moe-hot-cache-layer-curve` überall verfügbar. |
| `score_observations_for_arch(arch, observations, params)` | Wählt die Scoring-Strategie. Aktuell Qwen und Gemma über generisches Weighting. | Erweiterungspunkt für neue Modelle ohne Eingriff in den Loader. |
| `collect_expert_sizes(model)` | Sammelt alle Expert-Slices aus Layern mit `ffn_down_exps`. | Verhindert, dass nicht-MoE-Layer oder nicht vorhandene Tensoren in den Cache geraten. |
| `estimate_kv_cache_bytes_on_device(model, params, dev)` | Schätzt KV-Cache-Bedarf für `--moe-hot-cache-max-mib -1`. | Auto-Sizing kann VRAM für KV reservieren und Warmup-OOM vermeiden. |
| `auto_hot_cache_budget_bytes(model, params, dev, reserve_kv_cache)` | Liest freien VRAM, zieht KV-Reserve und Safety-Reserve ab. | `-1` nutzt möglichst viel übrigen VRAM, aber mit Sicherheitsabstand. |
| `llama_moe_hot_cache_parse_perf_json_observations(json)` | Validiert Schema, liest Layer, Expertenlisten und Timings. | Ein Parser für `/moe-layer-perf` und `--moe-layer-perf-out`. |
| `llama_moe_hot_cache_parse_perf_json(json)` | Gibt das Default-Hit-Ranking zurück. | Rückwärtskompatibler, einfacher Einstieg für Tests und ältere Nutzer. |
| `llama_moe_hot_cache_select(observed, sizes, budget)` | Geht das Ranking der Reihe nach durch und packt Experten ins Budget; pro aktivem Layer wird ein Dummy-Expert mitgerechnet. | Budget wird nicht überschritten; Layer mit Cache bekommen einen sicheren Dummy-Slot. |
| `llama_moe_hot_cache_init(model, params, reserve_kv_cache)` | Voller Cache-Aufbau: JSON lesen, scoren, Budget berechnen, ggml-Kontext anlegen, Cache-Tensoren erzeugen, Expert-Slices kopieren, Maps/Masks setzen. | Zentraler Einstieg für die ganze Funktion. Alle teuren Kopien passieren einmal beim Start oder beim Auto-Init. |
| `llama_moe_hot_cache_init_after_model_load(model, params)` | Baut den Cache direkt nach Modell-Load, wenn `max_mib > 0`. | Für festes Budget: Cache entsteht vor Context/KV. |
| `llama_moe_hot_cache_init_after_context_memory(model)` | Baut den Cache nach Context/KV, wenn `max_mib == -1`. | Für Auto-Sizing: erst Modell/KV laden, dann Rest-VRAM nutzen. |
| `llama_moe_hot_cache_layer_active(model, il)` | Prüft, ob der Layer einen aktiven Cache hat. | Qwen/Gemma können mit einem kleinen Hook zwischen Standard- und Hot-Graph wählen. |
| `llama_model_load_from_file_impl()` Hook in `src/llama.cpp` | Ruft `llama_moe_hot_cache_init_after_model_load()`. | Fester Cache wird ohne Modellklassen-Änderung initialisiert. |
| `llama_context` Konstruktor-Hook | Ruft `llama_moe_hot_cache_init_after_context_memory()`. | Auto-Cache nutzt den wirklich freien VRAM nach Context-Speicher. |

## Worklists

Dateien: `src/llama-moe-hot-cache.cpp`, `src/llama-moe-hot-cache-graph.cpp`

Die Worklist ist ein `F32`-Tensor mit Form:

```text
[capacity, LLAMA_MOE_HOT_CACHE_WORKLIST_FIELD_COUNT]
capacity = n_tokens * n_expert_used
```

Jedes Feld ist eine Zeile. Gültige Hot-Slots werden am Anfang der Hot-Felder dicht gepackt, gültige Cold-Slots am Anfang der Cold-Felder.

| Methode | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `llama_moe_hot_cache_build_worklist(dst, selected_experts, weights, layer, ith, nth)` | Nimmt fertige Top-K-Expert-IDs und Gewichte, schaut `layer.hot_id_map_host[expert]` nach und schreibt kompakte Hot/Cold-Listen inklusive Counts. | Trennt pro Token/Expert-Slot den GPU-Hot-Pfad vom CPU-Cold-Pfad. |
| `llama_moe_hot_cache_build_worklist_from_logits(dst, logits, layer, ith, nth)` | Berechnet für Decode (`n_tokens == 1`) Top-K und Softmax auf CPU direkt aus Logits und schreibt dieselbe Worklist. | Spart GPU-TopK/Weight-Knoten im Decode-Pfad und reduziert Routing-Overhead. |
| `llama_qwen35moe_hot_cache_build_worklist_op()` | `ggml_map_custom3`-Adapter für `llama_moe_hot_cache_build_worklist()`. | Bindet unsere C++-Logik als GGML-Node in den Graph ein. |
| `llama_qwen35moe_hot_cache_build_worklist_from_logits_op()` | `ggml_map_custom2`-Adapter für CPU-Routing direkt aus Logits. | Ermöglicht den schnelleren Decode-Routing-Pfad im Graph. |

Beispiel:

```text
selected_experts = [7, 12, 44, 101]
hot cache enthält 12 und 101

Hot-Liste:
  cache_id(12), src_slot 1, weight ...
  cache_id(101), src_slot 3, weight ...

Cold-Liste:
  expert 7, src_slot 0, weight ...
  expert 44, src_slot 2, weight ...
```

## Dynamisches Update

Datei: `src/llama-moe-hot-cache.cpp`

| Methode | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `llama_moe_hot_cache_update_from_perf_json(model, json, update_rate)` | Liest aktuelle Perf-Daten, berechnet Hit-Rate, ermittelt pro Layer Wunsch-Experten, bildet Evict/Add-Paare, sortiert nach Gain und tauscht maximal `ceil(update_rate * hot_experts)` Slots. | Der Cache passt sich nach jedem Request an, ohne Server-Neustart und ohne kompletten Cache-Rebuild. |
| `copy_expert_slice()` im Update-Kontext | Überschreibt den bestehenden Cache-Slot mit dem neuen Expert-Slice. | Der VRAM-Buffer bleibt gleich groß; nur Inhalte und Maps wechseln. |
| `set_tensor_i32_1d()` / `set_tensor_f32_1d()` im Update-Kontext | Aktualisiert `hot_id_map`, `hot_mask`, `cold_mask` für Evict/Add. | Graph-Routing sieht ab dem nächsten Request die neue Cache-Belegung. |

Der Update-Pfad verändert nicht die Anzahl Hot-Experten pro Layer. Er tauscht nur innerhalb bereits angelegter Slots. Dadurch bleibt die Graph- und Buffer-Form stabil.

## Weighting

Dateien: `src/models/qwen35moe-hot-cache.cpp`, `src/models/gemma4-hot-cache.cpp`

| Methode | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `str_is()` / `str_is_any()` | String-Vergleich für Mode-Namen. | Kleine, lokale Parser-Helfer ohne zusätzliche Dependencies. |
| `normalize_layer_curve(value)` | Begrenzung auf `0.0..1.0`, sonst Default `0.50`. | Verhindert extreme oder ungültige Gewichtung. |
| `layer_curve_from_env()` | Liest generische und alte Modell-spezifische Env-Namen. | Bewahrt Kompatibilität zu bisherigen Startscripten. |
| `total_hits(layer, expert)` | Nutzt `hot+cold` oder `raw`. | Weighting funktioniert mit Lernlauf und Hot-Cache-Lauf. |
| `score_to_u64(score)` | Rundet gewichtete Scores robust in `uint64_t`. | Ranking bleibt integer-basiert und sortierbar. |
| `layer_pressure(layer)` | Nutzt bevorzugt Join-Wait, sonst Lane-Delta, sonst Cold-Slots oder Wait-pro-Cold-Slot. | Langsame Cold-Layer können bei der Cache-Auswahl mehr VRAM bekommen. |
| `layer_pressure_for_source(layer, source)` | Wählt zwischen Parallel-Pressure und Gesamt-MoE-Zeit. | Unterstützt `pressure` und `time` mit derselben Infrastruktur. |
| `average_layer_pressure()` | Durchschnitt über Layer mit gültigem Druckwert. | Basis für relative Pressure-Gewichtung. |
| `minmax_layer_pressure()` | Min/Max über Druckwerte. | Basis für Time- und Smooth-Normalisierung. |
| `percentile_sorted()` | Interpoliertes Perzentil aus sortierten Werten. | Hilft beim robusten Glätten ohne Ausreißer-Dominanz. |
| `robust_layer_pressure_bounds()` | Nutzt 10. und 90. Perzentil als Druckgrenzen. | `smooth` reagiert weniger aggressiv auf einzelne Ausreißer. |
| `pressure_stats()` | Baut Durchschnitt und Bounds für eine Quelle. | Gemeinsame Vorberechnung für alle Pressure-Strategien. |
| `pressure_weight()` | Skaliert Layer relativ zum Durchschnitt mit begrenztem Min/Max. | Bevorzugt Layer, die am Join auf Cold warten. |
| `time_weight()` | Skaliert nach normalisierter Gesamt-MoE-Zeit. | Kann Layer bevorzugen, die insgesamt teuer sind. |
| `smooth_pressure_weight()` | Nutzt Wurzelkurve und begrenzten Boost. | Glättet starke Layer-Unterschiede. |
| `sort_entries()` | Deterministische Sortierung nach Score, Layer, Expert. | Reproduzierbare Cache-Listen. |
| `score_with_layer_weight()` | Multipliziert Expert-Hits mit Layer-Gewicht und gibt optional Hot-Sticky-Bonus. | Verbindet lokale Expert-Popularität mit globaler Layer-Kostenlage. |
| `qwen35moe_pressure_weighting::score()` | Scort mit Join/Cold-Pressure. | Optimiert auf den beobachteten CPU-Wartepfad. |
| `qwen35moe_smooth_pressure_weighting::score()` | Scort mit robust geglätteter Pressure. | Experimentierpfad für gleichmäßigere Layer-Verteilung. |
| `qwen35moe_time_weighting::score()` | Scort nach Gesamt-MoE-Zeit. | Alternative, wenn Join-Wait nicht stabil genug ist. |
| `qwen35moe_balanced_weighting::score()` | Rangiert Experten pro Layer und kombiniert Rank plus Hits. | Erzwingt stärker layerweise Verteilung. |
| `qwen35moe_flat_weighting::score()` | Interleaved Rank: erst bester Expert jedes Layers, dann zweitbester usw. | Default, weil Tests zeigten, dass gleichmäßigere Layer-Belegung bei begrenztem VRAM stabiler ist. |
| `weighting_strategy(mode)` | Gibt statische Strategieinstanz zurück. | Vermeidet Allocations und hält Mode-Auswahl zentral. |
| `llama_moe_hot_cache_weighting::parse_mode()` | Parst CLI/Env-Modi inkl. Aliase. | Nutzer können `flat`, `pressure`, `smooth`, `time`, `balanced` wählen. |
| `llama_moe_hot_cache_weighting::mode_name()` | Gibt kanonischen Namen zurück. | Logging und JSON bleiben lesbar. |
| `llama_moe_hot_cache_weighting::default_config()` | Baut Default aus Env; Default-Mode ist `flat`. | Beste bekannte Default-Verteilung ohne CLI-Pflicht. |
| `llama_moe_hot_cache_weighting::score_observations()` | Public Entry für Scoring. | Wird von Init und dynamischem Update genutzt. |
| `llama_moe_hot_cache_qwen35moe_weighting::*` | Thin Wrapper auf generisches Weighting. | Bewahrt alte Qwen-spezifische API-Namen. |
| `llama_moe_hot_cache_gemma4_weighting::score_observations()` | Nutzt generisches Weighting, erlaubt Gemma4-spezifische Layer-Curve-Env. | Gemma4 bekommt dieselben Strategien ohne Qwen-Code zu berühren. |

## Graph-Aufbau

Datei: `src/llama-moe-hot-cache-graph.cpp`

| Methode | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `llama_moe_hot_cache_graph_tweaks::parallel_mode()` | Liest `LLAMA_MOE_HOT_CACHE_PARALLEL`: `0/off`, `auto`, `force`. Default auto. | Hot/Cold-Parallelisierung kann ohne Rebuild gesteuert werden. |
| `parallel_min_slots()` | Liest Mindestzahl Slots für Parallelregion; Default `2`. | Verhindert Parallel-Overhead bei zu kleinen Workloads. |
| `merge_sum_rows()` | Schaltet Merge per `ggml_sum_rows`. | Schnellerer Merge als viele einzelne `ggml_add` bei mehreren Slots. |
| `cpu_decode_routing()` | Aktiviert CPU-Worklist direkt aus Logits im Decode. | Spart Routing-Knoten im Hot-Graph. |
| `decode_direct_merge()` | Aktiviert direkten Decode-Merge auf `[n_embd, 1]`. | Vermeidet vollständiges Slot-Tensor-Materialisieren. |
| `decode_strided_sum_rows()` | Erlaubt SumRows auf nicht-kontiguem Stride-Layout. | Spart `ggml_cont` und Kopien im Merge. |
| `hot_dummy_padding()` | Liest Dummy-Padding-Schalter. | Stabiler Umgang mit ungenutzten Hot-Slots. |
| `shared_input_row()` | Markiert Cold-Inputs als wiederverwendete erste Zeile bei Decode. | Reduziert Gather-Kosten im Cold-Pfad. |
| `cold_prefix_sum()` | Reduziert nur die gültigen Cold-Prefix-Slots. | Spart CPU-Arbeit, wenn wenige Cold-Slots aktiv sind. |
| `cold_prefix_weighted_sum()` | Integriert Gewichtung direkt in Prefix-Sum. | Spart separaten Mul-Knoten im Cold-Decode-Pfad. |
| `decode_repeat_hot_input()` | Wiederholt Hot-Input statt pro Slot zu gathern. | Kann Hot-Gather-Overhead im Decode reduzieren. |
| `cold_first_row_input()` | Kopiert nur die erste Input-Zeile für Cold-Decode. | Spezialfall für Decode mit gemeinsamem Input. |
| `branch_reduce_merge()` | Optionaler Branch-interner Reduce vor Join. | Experimenteller Pfad, besonders für Gemma4 als Vergleich/Fallback. |
| `env_enabled_by_default(name)` | Shared Env-Parser für boolsche Tweaks. | Alle Tweaks haben konsistente Default-Regel. |
| `llama_qwen35moe_hot_cache_graph_tweaks::get_profile()` | Qwen-Profil: direkte Decode-Optimierungen, aber `branch_reduce_merge` aus. | Qwen bleibt auf dem bewährten schnellen Pfad. |
| `llama_gemma4_hot_cache_graph_tweaks::get_profile()` | Gemma4-Profil: direkte Decode-Optimierungen und optional Branch-Reduce-Merge. | Gemma4 kann eigene Merge-Experimente nutzen, ohne Qwen zu beeinflussen. |
| `llama_moe_hot_cache_graph_profiles::profile_for_arch()` | Wählt Profil nach Architektur, default leer. | Neue Modelle müssen explizit opt-in machen; verhindert Seiteneffekte. |
| `llama_qwen35moe_hot_cache_sum_prefix_rows_op()` | Summiert nur die ersten `count` Zeilen eines Branch-Outputs. | Cold-Prefix-Merge reduziert CPU-Arbeit bei sparsamer Cold-Lane. |
| `llama_qwen35moe_hot_cache_sum_weighted_prefix_rows_op()` | Summiert Prefix-Zeilen mit Worklist-Gewichten. | Spart ein vorheriges Gewichts-Mul im Cold-Pfad. |
| `llama_qwen35moe_hot_cache_first_row_input_op()` | Kopiert die eine Decode-Input-Zeile in einen Branch-Tensor. | Hilft beim Shared-Input-Row-Pfad. |
| `llama_moe_hot_cache_set_mul_mat_id_flags(t, flags)` | Schreibt Flags in `op_params` von `ggml_mul_mat_id`. | Aktiviert unsere Kernel-Sonderfälle ohne neue GGML-Op. |
| `llama_moe_hot_cache_build_lora_mm_id(graph, w, cur, ids, flags)` | Baut `ggml_mul_mat_id` plus LoRA-Anteile mit denselben IDs/Flags. | Hot/Cold-Pfade bleiben LoRA-kompatibel. |
| `llama_moe_hot_cache_build_moe_ffn_with_ids(...)` | Baut den eigentlichen Expert-FFN für eine kompakte ID-Liste: Up/Gate/GateUp, Aktivierung, Down, Scale-Tensoren, Gewichtung, Output-Reduce. | Gemeinsamer Kern für Hot- und Cold-Branch, damit beide numerisch gleich zum normalen MoE-FFN bleiben. |
| `llama_moe_hot_cache_build_moe_hot_from_logits(graph, model, cur, logits, il, type_op)` | Generischer Hot-Cache-Layer aus bereits berechneten Router-Logits. Baut Worklist, Hot-Branch, Cold-Branch, Merge und Scheduler-Annotation. | Wird von Gemma4 genutzt und ist der wiederverwendbare Pfad für weitere Modelle. |
| `llama_model_qwen35moe::graph::build_layer_ffn_hot(cur, il)` | Qwen-spezifischer Hot-Layer. Berechnet Router-Logits selbst, baut Worklist, Hot/Cold-Branches und Merge. | Hält den Qwen-Pfad stabil und vermeidet Änderungen am Standard-Qwen-FFN außer dem kleinen Gate-Hook. |
| `llama_model_gemma4::graph::build_layer_moe_hot(cur, logits, il)` | Gemma4 ruft den generischen Hot-Pfad mit `LLM_FFN_GELU`. | Gemma4 bekommt Hot-Cache ohne Qwen-spezifische Kopie. |

## Modell-Hooks

Dateien: `src/models/qwen35moe.cpp`, `src/models/gemma4.cpp`, `src/models/models.h`

| Hook | Was er tut | Wie das hilft |
| --- | --- | --- |
| `llama_model_qwen35moe::graph::build_layer_ffn()` | Prüft `llama_moe_hot_cache_layer_active(model, il)` und ruft sonst den normalen `build_moe_ffn()`. | Bei deaktiviertem Hot-Cache bleibt Qwen im Upstream-Pfad. |
| `llama_model_gemma4::graph` MoE-Stelle | Berechnet Gemma-Router-Logits wie vorher und ruft bei aktivem Cache `build_layer_moe_hot()`. | Gemma4-Hot-Cache nutzt nur den MoE-Zweig und lässt Attention/MLP unverändert. |
| Deklarationen in `models.h` | Fügt `build_layer_ffn_hot()` und `build_layer_moe_hot()` zu den Modellgraph-Klassen hinzu. | Minimale Modellklassen-Schnittstelle für unsere separaten Hot-Cache-Dateien. |

## Perf-Erfassung und JSON

Dateien: `src/llama-moe-hot-cache-perf.cpp`, `src/llama-moe-hot-cache-perf.h`, `include/llama.h`

| Methode | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `llama_moe_layer_perf_local::ensure_shape_locked()` | Initialisiert Layer- und Expert-Zählergrößen. | Perf-Zähler passen sich dem aktuell geladenen Modell an. |
| `reset_locked(count_overflow)` | Setzt alle Zähler, Timings und Debug-Felder zurück. | Verhindert alte Daten nach Mode-Wechsel oder dynamischem Update. |
| `add_locked(dst, add)` | Saturating Add mit Reset bei Overflow. | Lange Läufe bleiben robust. |
| `add_expert_locked(layer, expert)` | Erhöht Raw-Expert-Hit und Gesamt-Hits. | Lernläufe ohne Hot/Cold-Split können Expert-Listen erzeugen. |
| `add_branch_expert_locked(layer, expert, hot)` | Erhöht Hot- oder Cold-Expert-Zähler. | UI und dynamisches Update sehen, was im letzten Lauf wirklich Hot oder Cold war. |
| `llama_moe_layer_perf_parse_mode(value, mode)` | Parst `full`, `update`, `off`. | Runtime-Schalter über HTTP und Env. |
| `llama_moe_layer_perf_mode_name(mode)` | Gibt JSON/Log-Namen zurück. | UI und Logs zeigen den aktiven Zählermodus. |
| `llama_moe_layer_perf_env_mode()` | Liest `LLAMA_MOE_LAYER_PERF`, default `full`. | Startverhalten kann per Env gesteuert werden. |
| `llama_moe_layer_perf_get_mode(ctx)` | Kombiniert Runtime-Mode, `--no-perf` und Env. | `--no-perf` kann Zähler wirklich abschalten. |
| `llama_moe_layer_perf_set_initial_mode(no_perf)` | Setzt Initialmodus beim Serverstart. | UI-Dropdown zeigt nach `--no-perf` korrekt `off`. |
| `llama_moe_layer_perf_set_mode(mode)` | Ändert Modus und resetet Snapshot/Zähler. | HTTP-POST kann Laufzeitkosten sichtbar reduzieren. |
| `llama_moe_layer_perf_is_enabled(ctx)` | True außer `off`. | Zentraler schneller Check. |
| `llama_moe_layer_perf_needs_expert_counts(no_perf)` | True für `full` und `update`. | Graph baut nur nötige Zählerknoten. |
| `llama_moe_parse_layer_from_name(name)` | Extrahiert Layer-ID aus Node-Namen wie `ffn_moe_hot_down-12`. | Eval-Callback kann Knoten dem Layer zuordnen. |
| `llama_moe_name_contains()` | String-Helfer für Node-Klassifikation. | Kleine, schnelle Prädikate statt komplexer Regex. |
| `llama_moe_is_*_node()` Prädikate | Erkennen TopK, Hot/Cold Counts, Gate/Up/Down, Routing, Merge, Gather/Scatter und Branch-Knoten. | Timings werden nach Ausführungsschritten sortierbar. |
| `llama_moe_layer_perf_begin(n_layer, n_expert, n_expert_used)` | Aktiviert Perf-Fenster für einen Graph-Compute. | Zähler messen nur echte Inferenz, nicht Warmup. |
| `llama_moe_layer_perf_end()` | Deaktiviert Perf-Fenster. | Verhindert versehentliches Zählen außerhalb des Graph-Laufs. |
| `llama_moe_layer_perf_reset()` | Public Reset. | Server kann nach dynamischem Update neu zählen. |
| `llama_moe_layer_perf_has_data()` | Prüft, ob verwertbare Daten existieren. | Server schreibt keine leere Perf-Datei. |
| `llama_moe_layer_perf_count_topk_locked()` | Liest TopK-IDs aus Backend-Tensor und zählt Raw-Experten. | Erstellt die initiale Expertenliste ohne Hot-Cache. |
| `llama_moe_layer_perf_count_worklist_count_locked()` | Liest Hot/Cold-Count aus Worklist. | Berechnet Hitrate und Slots pro Layer. |
| `llama_moe_layer_perf_count_branch_experts_locked()` | Liest Hot-Expert-IDs oder Cold-IDs und zählt pro Branch. | Dynamisches Update weiß, welche Experten aktiv waren. |
| `llama_moe_layer_perf_eval_callback(t, ask, user_data)` | GGML-Eval-Callback: fragt interessante Knoten ab, liest nach Ausführung IDs/Counts und misst Timings im Full-Mode. | Eine einzige Callback-Stelle liefert JSON, UI und Update-Daten. |
| `llama_moe_layer_perf_collect_parallel_metrics(sched)` | Holt Scheduler-Metriken für Parallelregionen und aggregiert sie pro Layer. | Zeigt Join-Wait, Overlap, Fallbacks und Lane-Zeiten. |
| `llama_moe_layer_perf_graph_compute_begin(ctx, sched)` | Aktiviert Scheduler-Perf, setzt Eval-Callback und startet Perf-Fenster. | Integriert die Messung in `llama_context::graph_compute()`. |
| `llama_moe_layer_perf_graph_compute_end(ctx, sched)` | Sammelt Scheduler-Metriken, beendet Perf und stellt den ursprünglichen Callback wieder her. | Verhindert Seiteneffekte auf andere llama.cpp Callback-Nutzer. |
| `llama_moe_layer_perf_json(ctx)` | Serialisiert Summary und Layer-Daten; im `off`-Mode minimale leere JSON. | API, UI, Lernfile und dynamisches Update nutzen dasselbe Format. |
| `llama_moe_layer_perf_json()` in `include/llama.h` | Öffentliche C-API für JSON. | Server kann ohne private Header auf die Daten zugreifen. |

Perf-Modi:

| Modus | Inhalt | Performance-Auswirkung |
| --- | --- | --- |
| `full` | Expert-Zähler, Hot/Cold-Zähler, alle Timings, Fallback-Debug. | Höchster Overhead, beste Analyse. |
| `update` | Nur Zähler, die dynamisches Update braucht, plus minimale Parallelzahlen. | Weniger Overhead, Update bleibt möglich. |
| `off` | Keine Zähler, minimale JSON. | Größter Speed-Gewinn. |

## Scheduler-Parallelisierung

Dateien: `ggml/include/ggml-backend-moe-hot-cache.h`, `ggml/src/ggml-backend-moe-hot-cache.inc`, `ggml/src/ggml-backend.cpp`

| Methode | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `ggml_backend_sched_moe_hot_cache_parallel_init(sched)` | Legt Scheduler-Erweiterungszustand an. | Separiert unsere Daten vom normalen Scheduler-Code. |
| `ggml_backend_sched_moe_hot_cache_parallel_free(sched)` | Beendet Worker-Thread und löscht Zustand. | Keine Thread-Leaks beim Scheduler-Free. |
| `ggml_backend_sched_moe_hot_cache_parallel_reset(sched)` | Leert annotierte Regionen vor neuem Graph. | Alte Regionen werden nicht auf neue Graphs angewendet. |
| `ggml_backend_sched_moe_hot_cache_parallel_is_split_boundary(sched, node)` | Erzwingt Split-Grenzen an Hot-Start, Cold-Start und Join. | Der Scheduler kann Hot- und Cold-Lanes als getrennte Split-Bereiche erkennen. |
| `ggml_backend_sched_moe_parallel_worker_loop(worker)` | Hintergrundthread rechnet Cold- oder Lane-Splitbereiche. | Kein Thread-Start pro Token; ein persistenter Worker senkt Overhead. |
| `ggml_backend_sched_moe_parallel_worker_new()` | Erstellt Worker und Thread. | Lazy init: nur bei aktivem Parallelpfad. |
| `ggml_backend_sched_moe_parallel_worker_free(worker)` | Stoppt und joint Worker. | Sauberes Shutdown-Verhalten. |
| `ggml_backend_sched_moe_parallel_worker_get(sched)` | Gibt vorhandenen Worker zurück oder legt ihn an. | Wiederverwendung über viele Decode-Schritte. |
| `ggml_backend_sched_moe_parallel_worker_start(...)` | Übergibt Splitbereich und Compute-State an Worker. | Cold-Lane kann parallel zur Hot-Lane laufen. |
| `ggml_backend_sched_moe_parallel_worker_wait(...)` | Wartet auf Worker-Fertigstellung und Status. | Join-Punkt synchronisiert korrekt. |
| `ggml_backend_sched_find_split_containing(sched, tensor)` | Sucht, in welchem Scheduler-Split ein Tensor-Knoten liegt. | Löst annotierte Tensoren in echte Split-Indizes auf. |
| `ggml_backend_sched_moe_parallel_split_backend_id()` | Gibt Backend-ID eines Splits zurück. | Debug- und Validierungsdaten. |
| `ggml_backend_sched_moe_parallel_fill_split_debug()` | Füllt Debug-Felder für Splits und Backends. | Erklärt Fallbacks in `/moe-layer-perf`. |
| `ggml_backend_sched_backend_is_cuda()` | Prüft Backend-Namen auf CUDA. | Hot-Lane wird nur auf CUDA parallelisiert. |
| `ggml_backend_sched_backend_is_cpu()` | Prüft Backend-Device-Typ CPU. | Cold-Lane wird nur als CPU-Lane akzeptiert. |
| `ggml_backend_sched_moe_parallel_record_fallback()` | Schreibt Fallback-Grund in Perf-Struktur. | UI/JSON zeigen, warum Parallelisierung nicht gegriffen hat. |
| `ggml_backend_sched_moe_parallel_fail_or_fallback()` | Bei `force`: Fehler; bei auto: einmal loggen und seriell weiter. | Sicheres Experimentieren ohne harte Crashes im Auto-Modus. |
| `ggml_backend_sched_resolve_moe_parallel_region()` | Validiert Split-Reihenfolge, Backends, Count-Prefix und Join. | Verhindert falsche parallele Ausführung bei anderem Graph-Layout. |
| `ggml_backend_sched_read_f32_count()` | Liest Hot/Cold-Count von CPU oder GPU. | Scheduler entscheidet, ob eine Lane leer ist. |
| `ggml_backend_sched_moe_parallel_auto_min_slots()` | Liest Mindest-Slots für Auto-Parallel. | Kleine Regionen laufen seriell, weil Parallel-Overhead sonst höher ist. |
| `ggml_backend_sched_zero_tensor(tensor)` | Nullt übersprungene Hot/Cold-Ausgabe bei leerer Lane. | Merge bleibt numerisch korrekt, auch wenn eine Lane nichts zu tun hat. |
| `ggml_backend_sched_compute_moe_parallel_region()` | Kern: prüft Slot-Counts, startet Cold-Worker, rechnet Hot-Lane lokal, synchronisiert beide, nullt leere Lanes, sammelt Perf. | Das ist der eigentliche Hot/Cold-Overlap. |
| `ggml_backend_sched_compute_splits(sched)` | Ersetzt den normalen Split-Compute für annotierte Regionen; dazwischen bleibt seriell. | Integration in den normalen GGML-Scheduler mit minimalem Hook. |
| `ggml_backend_sched_set_moe_hot_cache_parallel_perf_enabled()` | Schaltet Scheduler-Perf-Messung. | `--no-perf` und Runtime-Mode reduzieren Messoverhead. |
| `ggml_backend_sched_moe_hot_cache_parallel_region()` | Public API zum Annotieren einer Hot/Cold/Join-Region. | Graph-Code muss keine Scheduler-Interna kennen. |
| `ggml_backend_sched_get_moe_hot_cache_parallel_perf()` | Kopiert Scheduler-Perf-Daten heraus. | Perf-JSON kann Lane-Zeiten und Fallback-Gründe serialisieren. |
| Hook in `ggml-backend.cpp` | Inkludiert `.inc`, erweitert Scheduler-State, Split-Boundaries, Init/Free/Reset und Compute. | Möglichst wenig Core-Diff; Logik liegt in separater `.inc`-Datei. |

## GGML- und CUDA-Erweiterungen

Dateien: `ggml/src/ggml-cuda/*`, `ggml/src/ggml-cpu/*`, `ggml/include/ggml.h`

| Bereich | Was geändert wurde | Wie das hilft |
| --- | --- | --- |
| `ggml_mul_mat_id` Flags | `op_params` transportieren negative-ID-, Dummy- und Shared-Input-Row-Sonderfälle. | Hot/Cold-Graph kann ungültige Slots billig überspringen. |
| CUDA `mul_mat_id` | Interpretiert unsere Flags für Hot/Cold-Slots. | GPU-Pfad verarbeitet kompakte Hot-IDs ohne zusätzliche Umformung. |
| CPU `mul_mat_id` | Unterstützt die gleichen Flags im Cold-Pfad. | CPU-Cold-Lane bleibt numerisch kompatibel und kann ungültige Slots überspringen. |
| `ggml_cuda_sum_rows_utils::launch_f32_maybe_strided()` | Wählt contiguous oder strided `sum_rows` CUDA-Kernel. | Direct-Merge kann `ggml_cont` sparen und schneller summieren. |
| `reduce_rows_f32_strided` Sync-Anpassung | Fügt PDL-Synchronisation im strided Reduce-Pfad ein. | Stabilisiert CUDA-Reduce bei unserem strided Merge. |
| `ggml_backend_sched_moe_hot_cache_parallel_is_split_boundary()` Hook | Erzwingt Splitgrenzen an markierten Tensoren. | Hot/Cold-Parallelregion kann zuverlässig auf Backend-Splits abgebildet werden. |

Diese Änderungen sind die konfliktträchtigsten Stellen bei Upstream-Rebases. Die eigentliche Feature-Logik liegt deshalb möglichst in `llama-moe-hot-cache-*` und `ggml-backend-moe-hot-cache.inc`.

## CLI- und Common-Parameter

Dateien: `common/arg.cpp`, `common/common.h`, `common/common.cpp`, `include/llama.h`

| Methode/Feld | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `common_params::moe_hot_cache_max_mib` | `0` aus, positive MiB fix, `-1` Auto-Sizing. | Ein Argument deckt statischen und automatischen Cache ab. |
| `common_params::moe_hot_cache_auto_reserve_mib` | VRAM-Sicherheitsreserve beim Auto-Sizing. | Verhindert Warmup- oder MTP-OOM bei knappem Speicher. |
| `common_params::moe_hot_cache` | Pfad zur Perf-JSON. | Entkoppelt Lernlauf und Cache-Lauf. |
| `common_params::moe_hot_cache_update_rate` | Anteil zu tauschender Cache-Slots nach Request. | Dynamischer Cache ohne Neustart. |
| `common_params::moe_hot_cache_layer_curve` | Stärke der Layer-Gewichtung. | Nutzer kann glatte oder aggressive Verteilung wählen. |
| `common_params::moe_hot_cache_weighting` | Modus `flat`, `pressure`, `smooth`, `time`, `balanced`. | Experimentieren ohne Codeänderung. |
| `common_params::moe_layer_perf_out` | Datei für Perf-JSON nach Request. | Erster Lernlauf braucht keinen HTTP-Abruf. |
| `llama_moe_hot_cache_set_layer_curve_env()` | Spiegelt CLI-Wert in Env. | Weighting-Code kann zentral aus Env/Params lesen. |
| `llama_moe_hot_cache_weighting_valid()` | Validiert CLI-Mode. | Fehler früh beim Start statt später im Cache-Init. |
| `llama_moe_hot_cache_set_weighting_env()` | Spiegelt CLI-Mode in Env. | Gleiches Verhalten in altem Env- und neuem Argumentpfad. |
| Parse-Validierungen nach `common_params_parse_ex()` | Erzwingen `--moe-hot-cache` bei `max_mib != 0`, `-1` nur mit `--ctx-size`, Update nur mit Cache. | Fehlkonfigurationen brechen vor Modellstart ab. |
| `common_model_params_from_common_params()` Erweiterung | Kopiert Hot-Cache-Parameter in `llama_model_params`. | Loader und Context haben alle nötigen Werte ohne globale State-Abhängigkeit. |
| Argument `--moe-hot-cache-max-mib` | Setzt Cache-Größe. | Hauptschalter für Feature-Aktivierung. |
| Argument `--moe-hot-cache-auto-reserve-mib` | Setzt Safety-Reserve. | Tuning gegen OOM. |
| Argument `--moe-hot-cache` | Setzt JSON-Datei. | Auswahlquelle für Hot-Experts. |
| Argument `--moe-hot-cache-update-rate` | Setzt dynamische Austauschrate. | Aktiviert Prompt-zu-Prompt-Anpassung. |
| Argument `--moe-hot-cache-layer-curve` | Setzt Layer-Kurve. | Gewichtung kann pro Test geändert werden. |
| Argument `--moe-hot-cache-weighting` | Setzt Weighting-Strategie. | Wechsel zwischen Flat, Pressure, Smooth, Time, Balanced. |
| Argument `--moe-layer-perf-out` | Schreibt JSON-Datei und deaktiviert `no_perf`. | Lernlauf kann direkt eine Expertenliste erzeugen. |

## Server-Integration

Dateien: `tools/server/server.cpp`, `tools/server/server-context.cpp`, `tools/server/server-context.h`, `tools/server/server-models.cpp`

| Methode/Hook | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `llama_moe_layer_perf_set_initial_mode(params.no_perf)` im Serverstart | Initialisiert Perf-Modus nach `--no-perf`. | UI und Backend starten konsistent. |
| Route `GET /moe-layer-perf` | Gibt aktuelle oder gespeicherte Perf-JSON zurück. | UI und externe Tools können live auslesen. |
| Route `POST /moe-layer-perf` | Setzt Runtime-Modus `full/update/off`. | Zähler können ohne Server-Neustart reduziert werden. |
| `remember_moe_layer_perf_json()` | Speichert letzten Snapshot. | Nach Reset/Update bleibt die letzte nutzbare Ansicht verfügbar. |
| `forget_moe_layer_perf_json()` | Löscht Snapshot. | Mode-Wechsel zeigt keine alten Daten. |
| `has_remembered_moe_layer_perf_json()` | Prüft gespeicherten Snapshot. | Hilft beim sicheren Schreiben von Dateien. |
| `get_moe_layer_perf_json()` | Gibt Live-Daten, Snapshot oder leere JSON zurück. | Einheitliche Quelle für Route, Datei und Update. |
| `write_moe_layer_perf_file()` | Schreibt `--moe-layer-perf-out`, wenn Daten existieren. | Lernworkflow erzeugt automatisch die JSON-Datei. |
| `set_moe_layer_perf_mode()` | Ruft Perf-Mode-Setter und löscht Snapshot. | Runtime-Schalter wird zentral angewendet. |
| `update_moe_hot_cache_if_pending()` | Nach Request: holt Perf-JSON, ruft dynamisches Update, loggt Hit-Rate und Austausch. | Der Server passt den Cache erst nach sauber beendetem Request an. |
| `moe_hot_cache_update_pending` | Flag nach Request-Ende. | Verhindert Updates mitten in der Inferenz. |
| Router `server_models_routes::get_moe_layer_perf()` | Leitet `/moe-layer-perf` an das laufende Modell weiter oder verlangt `?model=` bei Mehrdeutigkeit. | Routermode kann mehrere Modellinstanzen sauber adressieren. |
| Router `server_models_routes::post_moe_layer_perf()` | Leitet Mode-Wechsel an Zielmodell weiter. | UI-Dropdown funktioniert auch im Routermode. |

## UI

Dateien: `tools/ui/src/...`

| Methode/Komponente | Was sie tut | Wie das hilft |
| --- | --- | --- |
| `MoeLayerPerfService.get(model)` | Holt `/moe-layer-perf`, im Routermode mit `model` und `autoload=false`. | UI kann gezielt das laufende Modell abfragen. |
| `MoeLayerPerfService.setMode(mode, model)` | Sendet `POST /moe-layer-perf`. | Runtime-Schalter für `full/update/off`. |
| `modeFromResponse(data)` | Leitet UI-Modus aus JSON ab. | Dropdown bleibt synchron zu Backend. |
| `refreshMoePerfMode()` | Lädt aktuellen Modus für aktives Modell. | Startzustand des Dropdowns stimmt auch nach Reload. |
| `handleMoePerfModeChange(event)` | Optimistischer Mode-Wechsel mit Rollback bei Fehler. | Bedienung bleibt direkt, Fehler zerstören den alten Zustand nicht. |
| Button `ROUTES.MOE_LAYER_PERF` | Öffnet separate Layer-Perf-Seite neben dem Promptfeld. | Button bleibt erreichbar, auch wenn Token/s-Anzeige springt. |
| `clamp()` | Begrenzt Werte wie Update-Intervall. | UI akzeptiert nur 0.5 bis 3 Sekunden. |
| `countsToMap()` | Wandelt JSON-Counts in Maps. | Schneller Zugriff pro Expert in der Heatmap. |
| `addCounts()` | Addiert Count-Paare in eine Map. | Hot/Cold/Raw-Zähler können kombiniert werden. |
| `totalCountsForLayer()` | Baut Gesamt-Hits aus Hot+Cold oder Raw-Fallback. | Active-Delta funktioniert mit Lernlauf und Hot-Cache-Lauf. |
| `activeDeltaForLayer()` | Vergleicht aktuellen mit vorherigem Snapshot. | Gelbe Felder zeigen Experten, die seit dem letzten Poll aktiv waren. |
| `toViewLayers()` | Sortiert Layer und ergänzt Maps. | Rendert eine stabile UI-Struktur. |
| `resolveExpertCount()` | Nutzt `n_expert` oder höchste Expert-ID. | Heatmap-Größe funktioniert auch bei reduzierten JSONs. |
| `layerHitRate()` | Berechnet Layer-Hitrate aus Ratio oder Slots. | Graph und Karten zeigen korrekte Hitrates. |
| `resolveAverageHitRate()` | Nutzt Summary-Ratio oder Layer-Durchschnitt. | Top-Anzeige bleibt verfügbar bei verschiedenen JSON-Modi. |
| `buildGraphPoints()` | Erzeugt SVG-Punkte je Layer-Hitrate. | Hitrate-Verlauf über Layer wird sichtbar. |
| `numberLocale()` / `formatDecimal()` / `formatInteger()` / `formatPercent()` | Lokalisierte Zahlenformatierung. | Verhindert verwirrende Punkt/Komma-Darstellung. |
| `metricValue()` | Liest Summary-Felder typsicher. | Timings zeigen `n/a`, wenn ein Feld im Update/Off-Modus fehlt. |
| `formatMetricValue()` | Formatiert us- und Count-Werte. | Timings sind kompakt lesbar. |
| `formatMetricShare()` | Berechnet Anteil am `total_moe_time_per_call_us`. | Zeigt, wo Zeit im MoE-Pfad hingeht. |
| `handleBack()` | Nutzt Browser-History, sonst Startseite. | Rückpfeil landet wieder im vorherigen Chat. |
| `expertClass()` | Wählt Farbe: Hot rot, Cold blau, Active gelb, Idle muted. | Schnelle visuelle Prüfung der Cache-Belegung. |
| `expertTitle()` | Tooltip pro Expert. | Debug ohne zusätzliche Detailseite. |
| `expertLegendClass()` | Legendenfarben. | Konsistenz zwischen Grid und Legende. |
| `handleIntervalInput()` | Normalisiert Poll-Intervall. | Verhindert zu schnelles oder zu langsames Polling. |
| `refresh()` | Lädt Perf-Daten, merkt vorherigen Snapshot und Fehlerzustand. | Live-Ansicht aktualisiert sich ohne Reset-Button. |
| `timingMetricCard()` | Rendert ein Timing mit Tooltip und Anteil. | Timing-Erklärung direkt in der UI. |
| `timingMetricGroup()` | Gruppiert Timings nach Ausführungsreihenfolge. | Analyse folgt Routing, Parallelregion, Lanes, Sync, Merge. |

## Tests

Datei: `tests/test-moe-hot-cache.cpp`

| Test/Helfer | Was er prüft | Warum wichtig |
| --- | --- | --- |
| `require_impl()` / `require` | Minimaler Test-Assert. | Test bleibt ohne externes Framework klein. |
| `set_env_var()` / `scoped_env_var` | Temporäre Env-Änderungen. | Weighting-Defaults können isoliert getestet werden. |
| `test_default_weighting_is_flat()` | Default-Weighting ist `flat`. | Sichert aktuelle Standardentscheidung. |
| `test_parse_and_sort()` | JSON-Parsing und deterministische Sortierung. | Initialer Lernlauf wird korrekt verarbeitet. |
| `test_parse_branch_counts_and_layer_weight()` | `hot_experts`/`cold_experts` und Pressure-Scoring. | Update-Daten fließen korrekt in Auswahl ein. |
| `test_qwen_layer_pressure_uses_total_wait()` | Qwen-Pressure bevorzugt wartende Layer. | Regressionsschutz für Layer-Kurve. |
| `test_gemma4_layer_pressure_uses_total_wait()` | Gemma4 nutzt dieselbe Weighting-Infrastruktur. | Schützt Gemma4 vor Qwen-Speziallogik. |
| `test_flat_weighting_spreads_budget_over_layers()` | Flat verteilt Budget layerweise. | Sichert den Default gegen Rückfall zu reiner Hit-Sortierung. |
| `test_parse_raw_opt_schema()` | Neues optimiertes JSON-Schema wird gelesen. | `/moe-layer-perf` und `--moe-layer-perf-out` bleiben kompatibel. |
| `test_select_budget()` | Budget inklusive Dummy-Expert. | Verhindert VRAM-Überbuchung. |
| `test_bad_schema()` | Falsches Schema wirft Fehler. | Frühe Diagnose bei falscher Datei. |
| `make_ctx()` | Erstellt kleinen GGML-Kontext. | Worklist-Tests laufen ohne Modell. |
| `get_worklist_field()` | Liest ein Worklist-Feld. | Tests prüfen das genaue Packlayout. |
| `set_selected()` / `set_weight()` / `set_logit()` | Schreiben Testtensoren. | Worklist-Tests bleiben verständlich. |
| `require_close()` | Float-Vergleich. | Softmax-Gewichte aus Logits werden stabil geprüft. |
| `test_build_worklist_mixed()` | Gemischte Hot/Cold-Slots, Padding und Counts. | Wichtigster Test für Worklist-Korrektheit. |
| `test_build_worklist_all_hot_or_cold()` | Randfälle alle Hot oder alle Cold. | Verhindert ungültige Dummy- und Cold-IDs. |
| `test_build_worklist_from_logits()` | CPU-Decode-Routing aus Logits inklusive Softmax und Weight-Scale. | Schützt den schnelleren Decode-Routing-Pfad. |

## Dokumentations-Commits

Die Commits `5a6edbb43`, `2b4944906` und `27de9be40` enthalten keine neuen Laufzeitmethoden, dokumentieren aber wichtige Entscheidungen:

| Datei | Inhalt | Nutzen |
| --- | --- | --- |
| `README.md` | Quickstart, Build, Workflow, Break-even-Graph, Hinweise zu `--cpu-moe`, Auto-Sizing und UI. | Einstieg für Nutzer. |
| `docs/development/moe-hot-cache-developer-guide.md` | Deutscher Entwicklerguide zu Architektur, Workflow, Tuning und Experimenten. | Tiefer technischer Kontext. |
| `docs/development/moe-hot-cache-developer-guide.en.md` | Englische Version. | Extern teilbar. |
| `docs/development/moe-hot-cache-runtime-switches.md` | Argumente und Env-Variablen. | Nachschlagewerk für Startscripte und INI-Konfiguration. |
| `docs/development/moe-hot-cache-parallelization-history.md` | Historie der Performance-Hebel. | Warum bestimmte Pfade default an/aus sind. |
| `docs/development/moe-hot-cache-mtp-learnings*.md` | MTP-Ergebnisse und warum MTP lokal kein Default-Win war. | Verhindert, dass wir dieselben Experimente wiederholen. |
| `docs/development/moe-hot-cache-warm-lane-analysis.md` | Quadro-M1200/Warm-Lane-Analyse. | Dokumentiert, warum die zweite GPU in diesem Setup nicht lohnt. |

## Rebase-Risiko

Niedriges Risiko, weil Logik in eigenen Dateien liegt:

- `src/llama-moe-hot-cache.cpp`
- `src/llama-moe-hot-cache-graph.cpp`
- `src/llama-moe-hot-cache-perf.cpp`
- `src/models/qwen35moe-hot-cache.cpp`
- `src/models/gemma4-hot-cache.cpp`
- `ggml/src/ggml-backend-moe-hot-cache.inc`
- UI-Seite und Doku-Dateien

Höheres Risiko bei Upstream-Rebase:

- `ggml/src/ggml-backend.cpp`: Scheduler-Hooks und Include der `.inc`.
- `ggml/src/ggml-cuda/ggml-cuda.cu`: `mul_mat_id` Flag-Semantik.
- `ggml/src/ggml-cpu/*`: CPU-`mul_mat_id` Flag-Semantik.
- `ggml/src/ggml-cuda/sumrows.cu` und `reduce_rows.cuh`: strided `sum_rows`.
- `src/models/qwen35moe.cpp` und `src/models/gemma4.cpp`: kleine Hot-Cache-Gates im Modellgraph.
- `common/arg.cpp`: CLI-Argumente nahe Upstream-Argumentblöcken.

Beim nächsten Refactor sollten diese Core-Hooks möglichst dünn bleiben: Hook im Upstream-File, Logik in unseren separaten Dateien.
