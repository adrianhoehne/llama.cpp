# Dense Model TG Optimization Notes

Stand: 2026-05-23

Diese Notiz haelt die Analyse fuer dichte Modelle wie Qwen3.5 27B fest. Sie ist bewusst getrennt von der MoE-Hot-Cache-Dokumentation, weil Dense-Modelle keinen Experten-Selektionspfad haben und deshalb nicht direkt vom MoE-Experts-First-Mechanismus profitieren.

## Kurzfassung

Ja, bei Dense-Modellen kann man die Token-Generation wahrscheinlich verbessern, aber nicht mit dem gleichen Hot-Cache-Prinzip wie bei MoE.

Bei MoE-Modellen werden pro Token nur ausgewaehlte Experten aktiv. Dadurch kann man haeufig genutzte Experten in VRAM halten und seltene Experten auf CPU/RAM lassen. Bei Dense-Modellen muss dagegen jeder Layer und jeder FFN-Tensor bei jedem Token laufen. Es gibt keine Hot-Expert-Auswahl.

Der naheliegende Dense-Hebel ist deshalb nicht "welche Experten cache ich?", sondern "welche dichten Tensoren lege ich mit dem knappen VRAM am wirkungsvollsten auf die GPU?".

## Aktueller Codepfad fuer Qwen3.5 Dense

Qwen3.5 Dense laeuft ueber:

```text
src/models/qwen35.cpp
```

Der 27B-Fall wird dort ueber die Layerzahl erkannt. Qwen3.5 27B hat im lokalen Codepfad 64 Transformer-Layer.

Der Graph verarbeitet die Layer sequenziell:

```text
for each layer:
  attention oder recurrent/linear attention
  dense FFN
  residual / norm
```

Der Dense-FFN nutzt die klassischen Tensoren:

```text
blk.N.ffn_up.weight
blk.N.ffn_gate.weight
blk.N.ffn_down.weight
```

Die Matmuls gehen ueber `llm_graph_context::build_lora_mm(...)` in:

```text
src/llama-graph.cpp
```

und landen am Ende bei `ggml_mul_mat`.

## Warum `--n-gpu-layers` grob ist

Die Layer-Platzierung passiert generisch in:

```text
src/llama-model.cpp
```

`--n-gpu-layers` verschiebt ganze spaete Layer auf die GPU. Wenn der VRAM fuer einen weiteren ganzen Layer nicht reicht, bleibt dieser Layer komplett im CPU/RAM-Pfad, auch wenn einzelne Tensoren des Layers noch in den freien VRAM passen wuerden.

Das ist bei Dense-Modellen wahrscheinlich unguenstig, weil einzelne FFN-Tensoren gross und teuer sind. Ein kompletter Layer kann zu teuer fuer den VRAM sein, aber ein Teil des Layers koennte trotzdem lohnend sein.

## Beste naechste Idee: Dense-Tensor-Offload-Planer

Ein eigener Dense-Offload-Planer koennte nach dem normalen `ngl`-Offload den freien VRAM lesen und gezielt einzelne Tensoren auf CUDA0 legen.

Moegliche Startstrategie:

```text
1. Modellarchitektur erkennen, z.B. Qwen35 Dense.
2. Freies VRAM-Budget nach Modell-/Context-Load bestimmen.
3. Kandidatenliste aus CPU-residenten Dense-Tensoren bauen.
4. Zuerst FFN-Tensoren bevorzugen:
   - ffn_up.weight
   - ffn_gate.weight
   - ffn_down.weight
5. Solange Budget vorhanden ist, Tensoren per Tensor-Override auf CUDA0 legen.
6. Spaeter optional Attention-/Recurrent-Tensoren ergaenzen.
```

Der Nutzen waere eine feinere VRAM-Nutzung als `--n-gpu-layers`. Das Risiko sind zusaetzliche CPU/GPU-Grenzen innerhalb eines Layers. Deshalb muss man messen, ob gesparte CPU-Matmul-Zeit groesser ist als Transfer- und Scheduler-Kosten.

## Bestehender Mechanismus: `--override-tensor`

llama.cpp kann einzelne Tensoren bereits ueber `--override-tensor` auf ein bestimmtes Backend legen.

Beispiel:

```text
--override-tensor 'output\.weight=CPU'
--override-tensor 'blk\.(58|59|60|61)\.ffn_(up|gate|down)\.weight=CUDA0'
```

Die Layernummern muessen an das konkrete `ngl`-Setup angepasst werden. Der Test waere:

1. Baseline mit aktuellem `--n-gpu-layers`.
2. Output-Tensor testweise auf CPU legen, um VRAM freizugeben.
3. Mit dem gewonnenen VRAM einzelne FFN-Tensoren aus CPU-Layern auf CUDA0 legen.
4. TG, PP und Speicherverbrauch vergleichen.

## Output-Tensor als Tradeoff

Der Output-Tensor kann viel VRAM binden. Wenn `output.weight` auf CPU gelegt wird, passt eventuell ein weiterer Transformer-Layer oder mehrere FFN-Tensoren auf die GPU.

Das kann schneller oder langsamer sein:

- schneller, wenn die gewonnenen GPU-Layer mehr bringen als der CPU-Output kostet,
- langsamer, wenn der Output-Matmul dominant wird.

Deshalb ist das ein guter Benchmark-Kandidat, aber kein sicherer Default.

## Tensor-Splitting ueber mehrere GPUs

llama.cpp hat vorhandene Split-Modi:

```text
--split-mode row
--split-mode tensor
```

Qwen35 Dense ist im Code nicht explizit von Tensor-Splitting ausgeschlossen. Der praktische Haken ist, dass der CLI-Pfad fuer Tensor-Mode CPU-Geraete ablehnt. Tensor-Splitting ist also eher CUDA0+CUDA1 als CPU+GPU.

Auf einer langsamen zweiten GPU wie Quadro M1200 ist das nicht garantiert hilfreich. Fuer Dense-Modelle ist es aber testbarer als die MoE-Warm-Lane, weil es keinen Hot/Cold-Expert-Join im selben Sinne gibt.

## MTP / Speculative Decoding

Qwen35 hat im Code einen MTP-Pfad. Wenn das konkrete Dense-Modell MTP-Gewichte enthaelt und genug Speicher vorhanden ist, kann MTP sichtbare Tokens/s erhoehen.

Wichtig: MTP reduziert nicht die rohe Target-Model-Kosten pro Schritt. Es erzeugt zusaetzliche Draft-Tokens und gewinnt nur, wenn genug Drafts akzeptiert werden und der Zusatzspeicher nicht die bessere Modellplatzierung verhindert.

## Weniger vielversprechend: Generic op-offload fuer TG

Im CUDA-Backend gibt es einen Offload-Schwellenwert fuer `MUL_MAT`. Fuer Prompt Processing mit groesseren Batches kann das relevant sein. Bei Decode/TG ist der Batch typischerweise `1`.

Wenn man den Schwellenwert fuer TG aggressiv auf `1` setzt, koennte das dazu fuehren, dass CPU-residente Gewichtsmatrizen pro Token zur GPU bewegt werden. Das ist wahrscheinlich langsamer als direkt auf CPU zu rechnen.

## Groesserer Eingriff: CPU+GPU Tensor-Parallelismus

Theoretisch koennte man Dense-Matmul ueber CPU und GPU splitten. Praktisch waere das ein groesserer Eingriff:

- CLI erlaubt CPU im Tensor-Split aktuell nicht,
- Scheduler und Backend-Zuordnung muessten CPU+GPU als gemeinsames Tensor-Parallel-Ziel akzeptieren,
- Reduktion/Allreduce und Transfers koennen den Gewinn auffressen,
- Rebase-Konflikte waeren wahrscheinlicher.

Das ist deshalb kein guter erster Schritt.

## Empfohlener naechster Test

Ohne neuen Code:

```text
1. Baseline mit aktuellem Qwen3.5 27B Dense Start.
2. Gleicher Start mit output.weight auf CPU.
3. Gleicher Start mit einzelnen FFN-Tensoren der ersten CPU-Layer auf CUDA0.
4. Optional CUDA0+CUDA1 mit --split-mode tensor testen.
```

Mit Code:

```text
DenseOffloadPlanner
  - getrennt von qwen35.cpp
  - budgetbasiert
  - nur aktiv bei explizitem Argument
  - zuerst FFN-Tensoren
  - spaeter Perf-gesteuerte Tensor-Rangfolge
```

Moegliche Argumente:

```text
--dense-gpu-cache-max-mib <N|-1>
--dense-gpu-cache-profile qwen35
--dense-gpu-cache-output-policy keep|cpu|auto
```

Der erste sinnvolle Code-Hebel waere also ein kleiner, sauber gekapselter Dense-Tensor-Offload-Planer. Wenn man per `--override-tensor` keinen Gewinn sieht, sollte man den Codepfad nicht bauen.
