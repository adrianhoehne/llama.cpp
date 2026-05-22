# MoE Hot-Cache und MTP: Learnings

Stand: 2026-05-22

Diese Notiz fasst die Erkenntnisse aus den Experimenten mit Qwen3.6-35B-A3B
MTP und dem MoE-Hot-Cache zusammen. Sie ist absichtlich implementierungsnah
geschrieben, damit der experimentelle Code zurueckgesetzt und bei Bedarf spaeter
gezielt neu aufgebaut werden kann.

## Kurzfazit

MTP brachte in den lokalen Tests keinen klaren Vorteil fuer das Zielsetup. Mit
aktivem Hot-Cache ohne MTP wurden hoehere Tokenraten erreicht als mit MTP.

Beobachtete Richtung:

- Hot-Cache ohne MTP lag in frueheren Laeufen bei bis zu ca. `28 tk/s`.
- Ein MTP-Lauf aus dem Log lag bei `25.33 tk/s` Eval-Rate, trotz hoher Draft
  Acceptance von ca. `94%`.
- Ohne Hot-Cache, aber mit groben `override-tensor`-Layern, scheiterte MTP
  haeufig am zusaetzlichen MTP-Context-Speicher.

Die wichtigste praktische Erkenntnis ist: MTP ist nicht kostenlos. Es erzeugt
einen zweiten Context gegen das Target-Modell und benoetigt zusaetzliche CUDA
Compute-/PP-Buffer. In einem knappen 12-GB-VRAM-Setup kann dieser Zusatzspeicher
mehr schaden als die Draft-Acceptance hilft.

## Was MTP im llama.cpp-Pfad macht

Bei `--spec-type draft-mtp` wird kein vollstaendig separates Draft-Modell
geladen, wenn das Target-Modell selbst MTP-Layer enthaelt. Stattdessen wird nach
dem normalen Target-Context ein zweiter MTP-Context auf demselben Target-Modell
erstellt:

```text
creating MTP draft context against the target model
llama_init_from_model(model_tgt, cparams_mtp)
```

Dieser MTP-Context braucht eigenen Graph-/Compute-Speicher. In einem
fehlgeschlagenen Lauf wurde genau hier ein OOM ausgeloest:

```text
allocating 800.02 MiB on device 0: cudaMalloc failed: out of memory
failed to create MTP context
```

Wichtig: `--n-gpu-layers-draft auto` hilft fuer diesen MTP-Fall nur begrenzt,
weil kein separates Draft-Modell mit eigenem Layer-Offload geladen wird. Der
kritische Speicher ist der zusaetzliche Context/Compute-Speicher.

## Warum MTP ohne Hot-Cache nicht geladen hat

Der Startbefehl ohne Hot-Cache nutzte grobe Tensor-Overrides:

```text
blk.(0|1|2|3|4|5|6|40).ffn_...=CUDA0
```

Damit lagen sieben fruehe MoE-Layer plus der komplette MTP-Layer 40 als volle
Expert-Layer auf CUDA0. Das kann fuer normale Inferenz noch passen, laesst aber
nicht zwingend genug Reserve fuer den nachtraeglich erzeugten MTP-Context.

Auch wenn kein `--ctx-size` im Log stand, setzte `--fit` den Context automatisch:

```text
n_ctx_seq (107008)
```

Das ist also praktisch ein ca. 100k-Context, nur automatisch aus dem
Speicher-Fit abgeleitet. MTP kam danach noch oben drauf.

## Warum der MTP-Layer im Hot-Cache zuerst cold war

Der MTP-Layer fuer Qwen3.6-35B-A3B ist typischerweise Layer `40`, weil das Modell
40 normale Transformer-Layer plus einen NextN/MTP-Layer hat.

Der Hot-Cache-Pfad nutzt einen Layer nur dann, wenn dieser Layer beim
Cache-Aufbau aktiv befuellt wurde:

```cpp
llama_moe_hot_cache_layer_active(model, il)
```

Wenn die Start-JSON keine Daten fuer Layer 40 enthaelt, kann der normale
Hot-Cache-Auswahlpfad keine Experten fuer Layer 40 auswaehlen. Das fuehrte im
Live-Snapshot zu:

```text
layer = 40
hot_slots_total = 0
cold_slots_total = 76384
hot_experts_count = 0
cold_experts_count = 254
```

Das Auto-Update konnte diesen Zustand nicht reparieren, weil es nur aktive
Hot-Cache-Layer aktualisiert. Ein komplett inaktiver Layer hat keine
Hot-Expert-Slots, die ausgetauscht werden koennen.

## Verworfener Ansatz: Random-Fallback

Ein kurzer Ansatz war, fehlende Layer ohne Perf-Daten zufaellig mit Experten zu
befuellen. Das wurde wieder entfernt.

Warum nicht behalten:

- Random-Auswahl macht Ergebnisse schwer reproduzierbar.
- Sie verschleiert, ob eine Perf-JSON wirklich Layer-Daten enthaelt.
- Fuer Rebase/Review ist ein zufaelliger Fallback schwerer zu begruenden.

Die bessere Variante ist ein expliziter MTP-Priority-Pfad, der nur fuer erkannte
MTP-Layer greift und deterministisch arbeitet.

## Besserer Ansatz: MTP-Layer explizit priorisieren

Fuer MTP wurde ein Ratio-Konzept getestet:

```text
--moe-hot-cache-mtp-layer-ratio N
```

Semantik:

- `0.0` = keine MTP-Sonderbehandlung
- `0.9` = ca. 90 Prozent der MTP-Experten priorisieren
- `1.0` = ganzen MTP-Layer priorisieren
- nicht gesetzt = Auto
  - mit `draft-mtp`: `1.0`
  - ohne `draft-mtp`: `0.0`

Bei `n_expert = 256` bedeutet:

```text
ratio 0.9 -> ceil(256 * 0.9) = 231 Experten
ratio 1.0 -> 256 Experten
```

Eine Ratio unter `1.0` ist sinnvoll, wenn einige MTP-Experten praktisch tot
sind. Dann bleiben Hot-Cache-Slots fuer andere Layer frei.

## Reimplementierung: benoetigte Parameter

In `common_params` sollte ein Auto-Sentinel genutzt werden:

```cpp
float moe_hot_cache_mtp_layer_ratio = -1.0f; // -1 = auto
```

Im CLI-/INI-Parser:

```text
--moe-hot-cache-mtp-layer-ratio N
LLAMA_ARG_MOE_HOT_CACHE_MTP_LAYER_RATIO=N
```

Validierung:

```cpp
if (value < 0.0f || value > 1.0f) {
    throw std::invalid_argument("--moe-hot-cache-mtp-layer-ratio must be between 0.0 and 1.0");
}
```

Beim Uebersetzen von `common_params` nach `llama_model_params`:

```cpp
const bool mtp_enabled =
    std::find(params.speculative.types.begin(),
              params.speculative.types.end(),
              COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

mparams.moe_hot_cache_mtp_layer_ratio =
    params.moe_hot_cache_mtp_layer_ratio >= 0.0f
        ? params.moe_hot_cache_mtp_layer_ratio
        : (mtp_enabled ? 1.0f : 0.0f);
```

In `llama_model_params` braucht es das Feld:

```cpp
float moe_hot_cache_mtp_layer_ratio;
```

Default in `llama_model_default_params()`:

```cpp
/*.moe_hot_cache_mtp_layer_ratio =*/ 0.0f,
```

## Reimplementierung: MTP-Layer erkennen

Fuer Qwen35MoE/Qwen3.6-MoE kann der MTP-Layer ueber
`nextn_predict_layers` erkannt werden.

Pseudocode:

```cpp
static std::vector<uint32_t> mtp_hot_cache_layers(const llama_model & model) {
    std::vector<uint32_t> layers;

    if (model.arch != LLM_ARCH_QWEN35MOE ||
        model.hparams.nextn_predict_layers == 0 ||
        model.hparams.nextn_predict_layers >= model.hparams.n_layer) {
        return layers;
    }

    const uint32_t first_mtp_layer =
        model.hparams.n_layer - model.hparams.nextn_predict_layers;

    for (uint32_t il = first_mtp_layer; il < model.hparams.n_layer; ++il) {
        if (model.layers[il].ffn_down_exps != nullptr) {
            layers.push_back(il);
        }
    }

    return layers;
}
```

Das ist bewusst enger als ein allgemeiner Random-Fallback. Nur echte
MTP-MoE-Layer werden bevorzugt.

## Reimplementierung: Auswahl mit MTP-Prioritaet

Beim initialen Hot-Cache-Aufbau sollte vor der normalen Auswahl eine
Prioritaetsliste erzeugt werden:

```cpp
struct priority_layer {
    uint32_t layer;
    size_t target_experts;
    size_t total_experts;
    float ratio;
};
```

Zielberechnung:

```cpp
target = min(total, ceil(total * ratio));
```

Auswahlstrategie:

1. Alle beobachteten Experten des priorisierten Layers zuerst nehmen, sortiert
   nach normalem Score aus der Perf-JSON.
2. Wenn weniger als `target` vorhanden sind, deterministisch aus den echten
   Modell-Expert-Sizes auffuellen.
3. Erst danach die nicht-priorisierten Layer nach normalem Ranking anhaengen.
4. Den bestehenden Budget-Selektor weiterverwenden.

Pseudocode:

```cpp
for entry in observed:
    if entry.layer is priority and selected[layer] < target[layer]:
        add(entry, max_score)

for size in expert_sizes:
    if size.layer is priority and selected[layer] < target[layer]:
        add({ size.layer, size.expert, max_score })

for entry in observed:
    if entry.layer is not priority:
        add(entry)

plan = select_by_budget(prioritized, expert_sizes, budget_bytes)
```

Wichtig: Das deterministische Auffuellen ist kein globaler Fallback. Es gilt nur
fuer explizit priorisierte MTP-Layer.

Erwartete Logzeile:

```text
MTP hot-cache priority: layer 40 selected 231/256 experts (target = 231, ratio = 90.00%)
```

## Reimplementierung: MTP-Graph an Hot-Cache anbinden

Der MTP-Graph braucht fuer den MoE-FFN einen schmalen Hook. Ziel ist, in
`qwen35moe.cpp` moeglichst wenig Code zu hinterlassen, um Rebase-Konflikte zu
reduzieren.

Minimaler Eingriff in `qwen35moe.cpp`:

```cpp
ggml_tensor * moe_out = build_layer_ffn(cur, il);
cb(moe_out, "mtp_ffn_moe_out", il);
```

Die eigentliche Implementierung kann in `llama-moe-hot-cache-graph.cpp` liegen:

```cpp
ggml_tensor * llama_model_qwen35moe::graph_mtp::build_layer_ffn(
        ggml_tensor * cur,
        const int il) {
    const auto & layer = model.layers[il];

    ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
    cb(logits, "ffn_moe_logits", il);

    if (llama_moe_hot_cache_layer_active(model, il)) {
        return llama_moe_hot_cache_build_moe_hot_from_logits(
            *this, model, cur, logits, il, LLM_FFN_SILU);
    }

    return build_moe_ffn(cur,
        nullptr,
        layer.ffn_up_exps,
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
        nullptr,
        n_expert, n_expert_used,
        LLM_FFN_SILU, true,
        hparams.expert_weights_scale,
        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
        logits, layer.ffn_gate_up_exps,
        layer.ffn_up_exps_s,
        layer.ffn_gate_exps_s,
        layer.ffn_down_exps_s);
}
```

Warum `logits` explizit erzeugen:

- Der Hot-Cache-Pfad braucht die Router-Logits, um Hot/Cold-Worklists aus den
  gleichen Gate-Ergebnissen zu bauen.
- Der Fallback-Pfad kann dieselben Logits an `build_moe_ffn` uebergeben, statt
  Gate-Logits erneut zu berechnen.

## Reimplementierung: MTP-Auto-Update

Wenn der MTP-Layer beim Start aktiv ist, kann das dynamische Update ihn wie
andere Layer aktualisieren. Das ist besonders bei Ratio `< 1.0` sinnvoll:

- Start mit `0.9` legt z.B. 231 Hot-Experten an.
- Nach dem Prompt koennen tote Experten innerhalb dieser 231 Slots gegen besser
  passende Experten getauscht werden.
- Die Layergroesse bleibt dabei gleich; das Update resized den Layer nicht.

Der Update-Pfad sollte MTP-Stats sammeln:

```cpp
bool mtp_layer_active;
bool mtp_layer_full;
uint32_t mtp_layer;
size_t mtp_hot_experts;
size_t mtp_total_experts;
size_t mtp_candidates;
size_t mtp_exchanged;
```

Server-Log nach Update:

```text
MoE hot-cache MTP layer update:
layer = 40, hot experts = 231/256, candidates = ..., exchanged = ...
```

Einschraenkung: Ein komplett inaktiver MTP-Layer kann durch das Auto-Update
nicht nachtraeglich aktiviert werden. Er muss beim Start mindestens einen
Hot-Cache-Eintrag bekommen.

## Speicher-Learnings

MTP benoetigt zusaetzliche Reserve. Fuer Auto-Sizing des Hot-Cache sollte bei
MTP konservativer reserviert werden:

```text
--moe-hot-cache-max-mib -1
--moe-hot-cache-auto-reserve-mib 1600
```

Der konkrete Wert ist hardware- und ctx-abhaengig. In den Logs waren `800 MiB`
fuer den MTP-Context-Compute-Buffer ein realer OOM-Ausloeser. `1024 MiB`
Reserve kann knapp sein, wenn zusaetzlich Warmup, Graph-Reserve und CUDA-
Transienten auftreten.

Ohne Hot-Cache, aber mit `override-tensor`, muss man bedenken:

- komplette MoE-Layer auf CUDA0 sind sehr grob,
- ein kompletter MTP-Layer 40 kostet viel mehr als eine Expertenauswahl,
- normale Inferenz kann passen, waehrend MTP danach trotzdem OOM geht.

## Performance-Learnings

Die hohe Draft-Acceptance alleine garantiert keinen Gewinn. Im beobachteten
MTP-Lauf:

```text
draft acceptance = 0.94107
eval time = 185305.21 ms / 4694 tokens = 25.33 t/s
```

Trotz ca. 94 Prozent Acceptance war die Endrate schlechter als gute Hot-Cache-
Laeufe ohne MTP. Wahrscheinliche Ursachen:

- zusaetzlicher MTP-Context erzeugt Speicher- und Graph-Druck,
- MTP-Layer selbst ist ein weiterer MoE-Layer,
- bei knappem VRAM muss der Hot-Cache kleiner werden,
- weniger Hot-Cache-Budget kann mehr Cold-Arbeit in anderen Layern erzeugen,
- die CPU/GPU-Balance des Hot/Cold-Pfads war ohne MTP bereits guenstiger.

Praktische Schlussfolgerung fuer dieses Setup:

- MTP ist fuer Qwen3.6-35B-A3B auf der RTX 2060 im lokalen Setup nicht der
  beste Default.
- Hot-Cache ohne MTP ist der bessere Performance-Pfad.
- MTP kann spaeter erneut getestet werden, wenn mehr VRAM verfuegbar ist oder
  wenn llama.cpp den MTP-Context-Speicher deutlich reduziert.

## Rebase- und Code-Isolations-Learnings

Fuer rebase-freundliche Reimplementierung:

- `common/speculative.cpp` nicht anfassen. Der lokale Fix wurde entfernt, weil
  dieser Bereich upstream wahrscheinlich selbst korrigiert wird.
- Keine breite Logik in `qwen35moe.cpp` lassen.
- `qwen35moe.cpp` nur mit einem schmalen Hook aendern:
  - `graph_mtp` speichert `model`,
  - MTP-MoE ruft `build_layer_ffn(cur, il)`.
- Die eigentliche Logik in eigene Hot-Cache-Dateien legen:
  - `llama-moe-hot-cache.cpp` fuer Auswahl/Budget/Update,
  - `llama-moe-hot-cache-graph.cpp` fuer Graph-Helfer.
- Kein Random-Fallback.
- MTP-Prioritaet streng an erkannte MTP-Layer binden.

Nicht vermeidbar, wenn MTP-Hot-Cache wirklich in den Graph soll:

- Eine kleine Deklaration in `models.h`.
- Ein kleiner Aufruf in `qwen35moe.cpp`.

## Empfohlene Ruecksetzung

Wenn der experimentelle MTP-Code verworfen wird, koennen die aktuellen tracked
Aenderungen in folgenden Dateien zurueckgesetzt werden:

- `common/arg.cpp`
- `common/common.cpp`
- `common/common.h`
- `include/llama.h`
- `src/llama-model.cpp`
- `src/llama-moe-hot-cache-graph.cpp`
- `src/llama-moe-hot-cache.cpp`
- `src/llama-moe-hot-cache.h`
- `src/models/models.h`
- `src/models/qwen35moe.cpp`
- `tools/server/server-context.cpp`

Untracked lokale Testdateien separat pruefen, nicht blind loeschen, wenn sie
noch Perf-Daten, Starter oder Modellkonfigurationen enthalten.

## Wenn MTP spaeter erneut getestet wird

Empfohlener Testplan:

1. Baseline ohne MTP messen.
2. MTP ohne Hot-Cache nur mit genug freiem VRAM testen.
3. MTP mit Hot-Cache und konservativer Reserve testen:

```text
--moe-hot-cache-max-mib -1
--moe-hot-cache-auto-reserve-mib 1600
--moe-hot-cache-mtp-layer-ratio 0.9
```

4. Nach dem ersten langen Prompt pruefen:

```text
/moe-layer-perf
```

Besonders relevant:

- Layer 40 `hot_slots_total`
- Layer 40 `cold_slots_total`
- Layer 40 `hot_experts_count`
- globaler Hot-Slot-Anteil
- End-to-End `eval t/s`
- MTP `draft acceptance`

Abbruchkriterium:

Wenn MTP trotz hoher Acceptance weniger `t/s` liefert als Hot-Cache ohne MTP,
sollte MTP fuer dieses Hardware-/Context-Setup deaktiviert bleiben.
