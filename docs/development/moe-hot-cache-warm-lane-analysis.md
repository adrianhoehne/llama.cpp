# MoE Hot Cache Warm-Lane Analysis

This note documents the Gemma 4 MoE experiment with a third "warm" cache lane on a secondary GPU.

## Setup

- Primary hot-cache device: `CUDA0`, RTX GPU
- Optional warm-cache device: `CUDA1`, Quadro M1200
- Cold branch: CPU
- Join/merge target: primary graph on `CUDA0`
- Model under test: `unsloth/gemma-4-26B-A4B-it-GGUF:Q6_K_XL`
- Context window during the run: `100000`
- The warm cache was filled from the same `/moe-layer-perf` data as the hot cache. No separate expert list was used.

The intended data flow was:

```text
hot experts  -> CUDA0 -> join/merge on CUDA0
warm experts -> CUDA1 -> bridge/sync -> join/merge on CUDA0
cold experts -> CPU   -> join/merge on CUDA0
```

## Runs

### 1. Broad Warm Lane

The first 3-lane run allowed CUDA1 to receive warm experts broadly.

```text
hot_slot_ratio        84.73 %
warm_slot_ratio        7.11 %
cold_slot_ratio        8.16 %

total_moe/call      1444.87 us
merge/call           861.29 us
parallel region      686.96 us
join wait            557.67 us

fallbacks                 0
```

Effective per-launch lane cost:

```text
hot lane     ~206 us
warm lane    ~830 us
cold lane    ~687 us
```

Result: CUDA1 was slower than CPU-Cold on average. Most warm traffic landed on layers where the M1200 did not pay for its transfer and sync cost.

### 2. Timing-Gated Warm Lane

The next implementation added a timing gate: warm candidates are only selected for layers where the previous run showed:

```text
parallel_cold_lane_wall_time_per_call_us > parallel_warm_lane_wall_time_per_call_us
```

That reduced warm to these layers:

```text
1, 2, 5, 7, 8, 9
```

Measured result:

```text
hot_slot_ratio        83.70 %
warm_slot_ratio        2.61 %
cold_slot_ratio       13.69 %

total_moe/call      1192.73 us
merge/call           742.55 us
parallel region      582.73 us
join wait            472.26 us

fallbacks                 0
```

This was a clear improvement over the broad warm lane:

```text
total_moe/call  1444.87 -> 1192.73 us  (-17.5 %)
merge/call       861.29 ->  742.55 us
region/call      686.96 ->  582.73 us
join wait        557.67 ->  472.26 us
```

However, after changing the distribution, the remaining CPU-Cold work changed too. In the new run the gated warm layers were no longer actually faster than CPU-Cold:

```text
Layer 1: warm 857 us, cold 497 us
Layer 2: warm 643 us, cold 433 us
Layer 5: warm 316 us, cold 201 us
Layer 7: warm 484 us, cold 268 us
Layer 8: warm 435 us, cold 326 us
Layer 9: warm 317 us, cold 273 us
```

Result: The gate removed the worst warm layers, but the remaining warm lane still cost more than it saved.

### 3. Warm Lane Disabled By Second-Stage Perf

The next run used the gated run as the new perf input. Since every warm-enabled layer was now slower than CPU-Cold, the timing gate selected no warm experts.

Measured result:

```text
hot_slot_ratio        83.49 %
warm_slot_ratio        0.00 %
cold_slot_ratio       16.51 %

total_moe/call      1085.95 us
merge/call           678.19 us
parallel region      525.13 us
join wait            423.56 us

warm_launches             0
fallbacks                 0
```

Improvement over the gated warm-lane run:

```text
total_moe/call  1192.73 -> 1085.95 us  (-9.0 %)
merge/call       742.55 ->  678.19 us
region/call      582.73 ->  525.13 us
join wait        472.26 ->  423.56 us
```

Improvement over the broad warm-lane run:

```text
total_moe/call  1444.87 -> 1085.95 us  (-24.8 %)
```

## Interpretation

The problem is not only that the Quadro M1200 is slower than the RTX. The bigger issue is that the warm lane adds a third synchronization and transfer path:

```text
CUDA1/Warm -> host/scheduler/bridge -> CUDA0/Join
CPU/Cold   -> host/scheduler        -> CUDA0/Join
CUDA0/Hot  -> CUDA0                 -> CUDA0/Join
```

So CUDA1 does not simply replace CPU work. It also adds:

- warm worker wait
- warm sync
- bridge graph execution
- bridge input transfer
- extra split/join scheduling
- final join pressure on CUDA0

Because the final merge must happen on CUDA0, the warm lane can also slow the cold path indirectly. If CUDA1 finishes late or forces extra bridge work, the join waits longer even when the CPU lane itself would have been good enough.

## Conclusion

For the tested Gemma 4 setup, the best measured state is:

```text
CUDA0 hot cache enabled
CPU cold branch enabled
CUDA1 warm cache disabled
```

The M1200 warm lane does not produce a stable speedup. Even when selected only for initially promising layers, the changed expert distribution makes the remaining CPU-Cold work faster than the CUDA1 warm path.

## Practical Rule

Only enable a secondary warm-cache GPU when it wins clearly after a second-stage validation run.

The useful decision flow is:

```text
1. Run without warm or with broad warm to collect timings.
2. Enable warm only for layers where warm < cold.
3. Run again with that gated warm plan.
4. Re-evaluate the gated run.
5. If warm is no longer faster than cold, disable warm.
```

For this hardware pair, step 5 is the current result.

## Follow-Up

The implementation should preserve the "warm disabled" decision when a perf file explicitly contains a completed run with:

```text
parallel_warm_launches = 0
warm_slot_ratio = 0
```

Otherwise, a restart from a no-warm perf file can fall back to broad warm selection because no warm-vs-cold comparison exists in that file.
