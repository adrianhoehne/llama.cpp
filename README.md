# Experimental MoE hot-cache fork

This fork adds an experimental MoE expert hot-cache path for llama.cpp. The goal is faster token generation for large MoE models on systems where the model has to be split between CPU/RAM and a smaller CUDA GPU.

Start here:

- [User guide and arguments](https://adrianhoehne.github.io/llama.cpp/docs/moe-hot-cache/moe-hot-cache-usage-guide.html)
- [Architecture explainer](https://adrianhoehne.github.io/llama.cpp/docs/moe-hot-cache/moe-hot-cache-architecture-explainer.html)
- [PP architecture](https://adrianhoehne.github.io/llama.cpp/docs/moe-hot-cache/moe-hot-cache-pp-architecture.html)
- [PP journey](https://adrianhoehne.github.io/llama.cpp/docs/moe-hot-cache/moe-hot-cache-pp-journey.html)
- [Journey and learnings](https://adrianhoehne.github.io/llama.cpp/docs/moe-hot-cache/moe-hot-cache-journey-learnings.html)
- [Interactive token visualization](https://adrianhoehne.github.io/llama.cpp/docs/moe-hot-cache/moe-experts-first-visual-explainer.html)
- [Qwen3Next implementation](https://adrianhoehne.github.io/llama.cpp/docs/moe-hot-cache/qwen3-coder-next-hot-cache-implementation.html)

The feature is experimental and workload-dependent. 
Supported:
- Qwen3.5/Qwen3.6 MoE models
- Gemma 4 26B-A4B
- Qwen3Next
- Mellum MoE models
- GPT-OSS MoE models
- DeepSeek2-family MoE models, including GLM Flash MoE GGUF exports
- Native GLM4 MoE models

## Cached-experts configuration examples

The examples below use `llama-server` and keep the normal model path on the primary CUDA device. Replace `MODEL.gguf` and `/path/to/moe-perf-data` with your model and the directory or JSON data produced for the hot-cache planner.

### Default cached-experts, one CUDA card

Use this when one CUDA device should hold the normal graph/KV path and as many cached experts as the automatic budget allows. Remaining experts stay on CPU via `--cpu-moe`.

```sh
LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING=1 \
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --model MODEL.gguf \
  --device CUDA0 \
  --split-mode none \
  --main-gpu 0 \
  --n-gpu-layers 99 \
  --cpu-moe \
  --ctx-size 4096 \
  --ubatch-size 32 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --moe-hot-cache /path/to/moe-perf-data \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache-auto-reserve-mib 1024 \
  --moe-hot-cache-pp-reduce-merge on
```

### Two CUDA cards

Use this when `CUDA0` should remain the primary card for graph/KV/router/final merge and `CUDA1` should act as an additional expert lane. `warm` fills the primary lane first, then the second lane. For similar cards, try `hot-even`.

```sh
GGML_CUDA_P2P=1 \
LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING=1 \
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --model MODEL.gguf \
  --device CUDA0 \
  --split-mode none \
  --main-gpu 0 \
  --n-gpu-layers 99 \
  --cpu-moe \
  --ctx-size 4096 \
  --ubatch-size 32 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --moe-hot-cache /path/to/moe-perf-data \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache-auto-reserve-mib 1024 \
  --moe-hot-cache-second-device CUDA1 \
  --moe-hot-cache-second-max-mib -1 \
  --moe-hot-cache-second-auto-reserve-mib 512 \
  --moe-hot-cache-device-strategy warm \
  --moe-hot-cache-pp-reduce-merge on
```

For a small primary GPU that should only run graph/KV/router/final merge, disable the primary expert cache and place experts on the second device:

```sh
GGML_CUDA_P2P=1 \
LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING=1 \
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --model MODEL.gguf \
  --device CUDA0 \
  --split-mode none \
  --main-gpu 0 \
  --n-gpu-layers 99 \
  --cpu-moe \
  --ctx-size 4096 \
  --ubatch-size 32 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --moe-hot-cache /path/to/moe-perf-data \
  --moe-hot-cache-max-mib 0 \
  --moe-hot-cache-second-device CUDA1 \
  --moe-hot-cache-second-max-mib -1 \
  --moe-hot-cache-second-auto-reserve-mib 512 \
  --moe-hot-cache-device-strategy warm \
  --moe-hot-cache-pp-reduce-merge on
```

### Three CUDA cards

Use this when `CUDA0` is the primary card and `CUDA1`/`CUDA2` are additional expert lanes. This is the intended shape for one primary GPU plus two expert-only GPUs. Keep per-device reserve high enough for temporary buffers; reduce `--ctx-size` or `--ubatch-size` first if CUDA allocation fails.

```sh
GGML_CUDA_P2P=1 \
LLAMA_MOE_HOT_CACHE_CPU_DECODE_ROUTING=1 \
LLAMA_MOE_HOT_CACHE_PARALLEL=1 \
./build/bin/llama-server \
  --model MODEL.gguf \
  --device CUDA0 \
  --split-mode none \
  --main-gpu 0 \
  --n-gpu-layers 99 \
  --cpu-moe \
  --ctx-size 4096 \
  --ubatch-size 32 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --moe-hot-cache /path/to/moe-perf-data \
  --moe-hot-cache-max-mib -1 \
  --moe-hot-cache-auto-reserve-mib 1024 \
  --moe-hot-cache-second-device CUDA1 \
  --moe-hot-cache-second-max-mib -1 \
  --moe-hot-cache-second-auto-reserve-mib 512 \
  --moe-hot-cache-third-device CUDA2 \
  --moe-hot-cache-third-max-mib -1 \
  --moe-hot-cache-third-auto-reserve-mib 512 \
  --moe-hot-cache-device-strategy hot-even \
  --moe-hot-cache-pp-reduce-merge on
```

Notes:

- `--moe-hot-cache-max-mib -1` auto-sizes a lane from currently free VRAM minus its reserve.
- `--moe-hot-cache-max-mib 0` disables the primary expert lane while keeping secondary or tertiary expert lanes available.
- `GGML_CUDA_P2P=1` enables CUDA peer-copy when the cards and driver support it; unsupported pairs fall back internally.
- `LLAMA_MOE_HOT_CACHE_PARALLEL=force` is a debugging mode for valid parallel regions. Use `1`/`auto` for normal runs.
- PP dense hot-cache is enabled by adapter profile for supported MoE paths. Use `LLAMA_MOE_HOT_CACHE_PP_DENSE=0` only for regression comparisons.
- Dense PP only starts at `--moe-hot-cache-pp-dense-min-tokens N`, default `256`; set it per model in `model_config.ini` when a profile needs a different threshold.
- `--moe-hot-cache-pp-min-hot-expert-ratio F` can keep low-coverage profiles on the normal PP path while still using the hot cache for TG.
- A general speedup claim for two GPUs could not be validated on the available test hardware because the cards are very asymmetric. Treat the two-GPU examples as configuration starting points, not as benchmark guidance.

These changes will probably never reach upstream llama because I broke the contribution rules hardly. I am a Java developer and the last time I wrote anything in C is, I even don't remember when it was, therefore, the bit of knowlegde of C that I had is gone. Secondly, this is a tool for me, I want it to function, I want it to be easy and I used other tools to create it faster.
And lastly, I saw some discussions in the PRs and the tone is not what I would expect. I know especially big PRs are hard to overlook, but great features often create big PRs. I also hate big PRs. But, sometimes they are necessary. Anyway, I don't want to have such discussions, it's just a waste of time.

I gave my best to steer the AI to not do bullshit. After the most work was done, I started a refactoring round to achive SoC as good as necessary. I am not done yet, but for now it is ok. The goal make it easier understandable and also create as less friction for rebases as possible.

Maybe, some time a good C programmer comes around, picks this idea and creates some small PRs without AI that follow the contribution rules. However, I'll try to keep this branch updated as good as possible.

## Benchmark results with RTX 2060 12GB VRAM & Quadro M1200 4GB VRAM

Run: `2026-06-19 16:41-22:44 +0200`

Config: `/home/adrian/llama.exp.data/model_config.ini`

Prompt: `/home/adrian/llama.exp.data/benchmark-it/prompts/two-turn-pp-1000tk.md`

Max tokens: `10000`

`PP t/s` is prompt-processing throughput. `TG t/s` is token-generation throughput. `Hot ratio` is `moe_hot_slot_ratio` from the benchmark result. `n-cpu-moe` is shown for non-hot-cache/default profiles when configured.

### Qwen3.6-35B-A3B, 34.66B total / 3B active

Origin: `unsloth/Qwen3.6-35B-A3B-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen3.6-35B-A3B-default-llama` | Q8_K_XL | 35.80 GiB | Default Llama | primary |  | 131072 | 34 | 56.23 | 17.31 | 0.00% | 8862 stop |  |
| `Qwen3.6-35B-A3B-rtx-hot` | Q8_K_XL | 35.80 GiB | Hot-Cache | primary |  | 131072 |  | 62.74 | 18.45 | 35.85% | 9472 stop | Low hot ratio limits the speedup. |
| `Qwen3.6-35B-A3B-rtx-primary-quadro-experts` | Q8_K_XL | 35.80 GiB | Hot-Cache | primary | secondary | 131072 |  | 59.99 | 11.79 | 47.79% | 10000 length | Higher hot ratio, but the asymmetric second GPU is slower here. |

### Hauhau Qwen3.5-122B-A10B, 122.11B total / 10B active

Origin: `HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive`

| Profile | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `hauhau-qwen35-122b-default-llama` | IQ3_XXS | 43.90 GiB | Default Llama | primary |  | 131072 | 43 | 35.75 | 4.90 | 0.00% | 8948 stop |  |
| `hauhau-qwen35-122b-rtx-hot` | IQ3_XXS | 43.90 GiB | Hot-Cache | primary |  | 131072 |  | 36.13 | 5.56 | 30.23% | 10000 length |  |
| `hauhau-qwen35-122b-rtx-primary-quadro-experts` | IQ3_XXS | 43.90 GiB | Hot-Cache | primary | secondary | 131072 |  | 30.97 | 4.02 | 42.67% | 10000 length | Asymmetric dual-GPU setup is slower than RTX-only in this run. |

### Qwen3-Coder-Next, 79.67B total / 3B active

Origin: `unsloth/Qwen3-Coder-Next-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `qwen3-coder-next-default-llama` | IQ4_NL | 36.53 GiB | Default Llama | primary |  | 131072 | 45 | 70.25 | 9.54 | 0.00% | 7010 stop |  |
| `qwen3-coder-next-rtx-hot` | IQ4_NL | 36.53 GiB | Hot-Cache | primary |  | 131072 |  | 68.37 | 11.31 | 42.74% | 6368 stop |  |
| `qwen3-coder-next-rtx-primary-quadro-experts` | IQ4_NL | 36.53 GiB | Hot-Cache | primary | secondary | 131072 |  | 53.78 | 6.08 | 55.26% | 7778 stop | Asymmetric dual-GPU setup is slower than RTX-only in this run. |

### Mellum2-12B-A2.5B Thinking, 12.15B total / 2.5B active

Origin: `local GGUFs on /media/seagate`

| Profile | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `mellum2-default-llama` | Q8_0 | 12.03 GiB | Default Llama | primary |  | 131072 | 8 | 287.26 | 34.62 | 0.00% | 5398 stop |  |
| `mellum2-rtx-hot` | Q8_0 | 12.03 GiB | Hot-Cache | primary |  | 131072 |  | 327.12 | 58.66 | 95.38% | 5842 stop |  |
| `mellum2-q4-default-llama` | Q4_K_M | 7.51 GiB | Default Llama | primary |  | 131072 |  | 1454.78 | 104.56 | 0.00% | 7486 stop | Good example when not to use this branch: if the model fits in VRAM, hot-cache overhead is too large. |
| `mellum2-q4km-rtx-hot` | Q4_K_M | 7.51 GiB | Hot-Cache | primary |  | 131072 |  | 1122.83 | 98.73 | 100.00% | 2299 stop |  |
| `mellum2-q6k-rtx-hot` | Q6_K | 10.13 GiB | Hot-Cache | primary |  | 131072 |  | 483.04 | 69.62 | 98.53% | 5271 stop |  |
| `mellum2-rtx-primary-quadro-experts` | Q8_0 | 12.03 GiB | Hot-Cache | primary | secondary | 131072 |  | 203.76 | 19.75 | 100.00% | 5701 stop | All observed experts fit in hot-cache, but the second GPU lane is the bottleneck. |

### Mellum2-12B-A2.5B Base, 12.15B total / 2.5B active

Origin: `local GGUF on /media/seagate`

| Profile | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `mellum2-base-rtx-hot` | BF16 | 22.64 GiB | Hot-Cache | primary |  | 131072 |  | 82.38 | 22.49 | 75.81% | 6057 stop |  |

### GPT-OSS-20B, 20.91B total / 3.6B active

Origin: `unsloth/gpt-oss-20b-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `gpt-oss-default-llama` | Q8_K_XL | 12.28 GiB | Default Llama | primary |  | 131072 | 45 | 115.25 | 13.97 | 0.00% | 7547 stop |  |
| `gpt-oss-rtx-hot` | Q8_K_XL | 12.28 GiB | Hot-Cache | primary |  | 131072 |  | 229.26 | 32.17 | 78.07% | 8207 stop |  |
| `gpt-oss-rtx-primary-quadro-experts` | Q8_K_XL | 12.28 GiB | Hot-Cache | primary | secondary | 131072 |  | 147.67 | 16.29 | 98.33% | 7303 stop |  |

### GLM-4.7-Flash, 29.94B total / active parameter count not specified

Origin: `unsloth/GLM-4.7-Flash-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `glm4-flash-default-llama` | Q8_K_XL | 33.17 GiB | Default Llama | primary |  | 131072 | 46 | 50.12 | 9.78 | 0.00% | 8167 stop |  |
| `glm4-flash-rtx-hot` | Q8_K_XL | 33.17 GiB | Hot-Cache | primary |  | 131072 |  | 50.49 | 9.76 | 11.20% | 9686 stop | Very low hot ratio; hot-cache does not improve this profile. |
| `glm4-flash-rtx-primary-quadro-experts` | Q8_K_XL | 33.17 GiB | Hot-Cache | primary | secondary | 131072 |  | 48.53 | 7.57 | 31.75% | 5870 stop |  |

### GLM-4.7-Flash-REAP-23B-A3B, 23.00B total / 3B active

Origin: `unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `glm4moe-default-llama` | Q8_K_XL | 25.63 GiB | Default Llama | primary |  | 131072 | 45 | 65.65 | 9.89 | 0.00% | 7629 stop |  |
| `glm4moe-rtx-hot` | Q8_K_XL | 25.63 GiB | Hot-Cache | primary |  | 131072 |  | 64.70 | 9.84 | 12.12% | 8615 stop | Very low hot ratio; hot-cache does not improve this profile. |
| `glm4moe-rtx-primary-quadro-experts` | Q8_K_XL | 25.63 GiB | Hot-Cache | primary | secondary | 131072 |  | 62.50 | 7.95 | 33.93% | 5627 stop |  |

### Gemma-4-26B-A4B-it, 25.23B total / 4B active

Origin: `unsloth/gemma-4-26B-A4B-it-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `gemma4-default-llama` | Q6_K_XL | 21.68 GiB | Default Llama | primary |  | 131072 | 22 | 91.77 | 17.26 | 0.00% | 10000 length |  |
| `gemma4-rtx-hot-cache` | Q6_K_XL | 21.68 GiB | Hot-Cache | primary |  | 131072 |  | 104.32 | 25.62 | 69.71% | 8085 stop |  |
| `gemma4-rtx-primary-quadro-experts` | Q6_K_XL | 21.68 GiB | Hot-Cache | primary | secondary | 131072 |  | 98.16 | 14.61 | 85.92% | 5684 stop | Higher hot ratio, but the asymmetric second GPU is slower here. |

## Benchmark results with RTX 3060 12GB VRAM & Quadro M1200 4GB VRAM

Run: `2026-06-20 15:51-21:16 +0200`

Config: `/home/adrian/llama.exp.data/model_config.ini`

Prompt: `/home/adrian/llama.exp.data/benchmark-it/prompts/two-turn-pp-1000tk.md`

Max tokens: `10000`

`PP t/s` is prompt-processing throughput. `TG t/s` is token-generation throughput. `Hot ratio` is `moe_hot_slot_ratio` from the benchmark result. `n-cpu-moe` is shown for non-hot-cache/default profiles when configured. The Qwen3.6 hot-cache profiles used `moe-hot-cache-auto-reserve-mib=1256`; no profile required `1512`.

### Qwen3.6-35B-A3B, 34.66B total / 3B active

Origin: `unsloth/Qwen3.6-35B-A3B-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 3060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `Qwen3.6-35B-A3B-default-llama` | Q8_K_XL | 35.80 GiB | Default Llama | primary |  | 131072 | 34 | 57.48 | 18.29 | 0.00% | 10000 length |  |
| `Qwen3.6-35B-A3B-rtx-hot` | Q8_K_XL | 35.80 GiB | Hot-Cache | primary |  | 131072 |  | 63.00 | 19.31 | 34.83% | 10000 length | Ran with `moe-hot-cache-auto-reserve-mib=1256`; no OOM. Low hot ratio limits the speedup. |
| `Qwen3.6-35B-A3B-rtx-primary-quadro-experts` | Q8_K_XL | 35.80 GiB | Hot-Cache | primary | secondary | 131072 |  | 59.88 | 12.69 | 48.59% | 10000 length | Ran with `moe-hot-cache-auto-reserve-mib=1256`; no OOM. Higher hot ratio, but the asymmetric second GPU is slower here. |

### Hauhau Qwen3.5-122B-A10B, 122.11B total / 10B active

Origin: `HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive`

| Profile | Quantization | Disk size | Mode | RTX 3060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `hauhau-qwen35-122b-default-llama` | IQ3_XXS | 43.90 GiB | Default Llama | primary |  | 131072 | 43 | 36.44 | 5.27 | 0.00% | 10000 length |  |
| `hauhau-qwen35-122b-rtx-hot` | IQ3_XXS | 43.90 GiB | Hot-Cache | primary |  | 131072 |  | 36.99 | 6.83 | 37.94% | 6302 stop | RTX-only hot-cache improves TG despite modest hot ratio. |
| `hauhau-qwen35-122b-rtx-primary-quadro-experts` | IQ3_XXS | 43.90 GiB | Hot-Cache | primary | secondary | 131072 |  | 30.97 | 4.31 | 43.50% | 8282 stop | Asymmetric dual-GPU setup is slower than RTX-only in this run. |

### Qwen3-Coder-Next, 79.67B total / 3B active

Origin: `unsloth/Qwen3-Coder-Next-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 3060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `qwen3-coder-next-default-llama` | IQ4_NL | 36.53 GiB | Default Llama | primary |  | 131072 | 45 | 69.96 | 10.18 | 0.00% | 7649 stop |  |
| `qwen3-coder-next-rtx-hot` | IQ4_NL | 36.53 GiB | Hot-Cache | primary |  | 131072 |  | 68.71 | 11.76 | 39.46% | 7124 stop |  |
| `qwen3-coder-next-rtx-primary-quadro-experts` | IQ4_NL | 36.53 GiB | Hot-Cache | primary | secondary | 131072 |  | 54.15 | 6.39 | 54.67% | 7124 stop | Asymmetric dual-GPU setup is slower than RTX-only in this run. |

### Mellum2-12B-A2.5B Thinking, 12.15B total / 2.5B active

Origin: `local GGUFs on /media/seagate`

| Profile | Quantization | Disk size | Mode | RTX 3060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `mellum2-default-llama` | Q8_0 | 12.03 GiB | Default Llama | primary |  | 131072 | 8 | 291.64 | 37.66 | 0.00% | 7475 stop |  |
| `mellum2-rtx-hot` | Q8_0 | 12.03 GiB | Hot-Cache | primary |  | 131072 |  | 339.21 | 64.95 | 95.06% | 2111 stop |  |
| `mellum2-q4-default-llama` | Q4_K_M | 7.51 GiB | Default Llama | primary |  | 131072 |  | 1608.33 | 118.63 | 0.00% | 6943 stop | This is a good example when NOT to use this branch. If the model can live completely in VRAM then the overhead is too big. |
| `mellum2-q4km-rtx-hot` | Q4_K_M | 7.51 GiB | Hot-Cache | primary |  | 131072 |  | 1223.02 | 110.24 | 100.00% | 5785 stop |  |
| `mellum2-q6k-rtx-hot` | Q6_K | 10.13 GiB | Hot-Cache | primary |  | 131072 |  | 509.10 | 77.65 | 98.41% | 2463 stop |  |
| `mellum2-rtx-primary-quadro-experts` | Q8_0 | 12.03 GiB | Hot-Cache | primary | secondary | 131072 |  | 207.19 | 20.69 | 100.00% | 7033 stop | All observed experts fit in hot-cache, but the second GPU lane is the bottleneck. |

### Mellum2-12B-A2.5B Base, 12.15B total / 2.5B active

Origin: `local GGUF on /media/seagate`

| Profile | Quantization | Disk size | Mode | RTX 3060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `mellum2-base-rtx-hot` | BF16 | 22.64 GiB | Hot-Cache | primary |  | 131072 |  | 103.19 | 24.69 | 75.69% | 6626 stop |  |

### GPT-OSS-20B, 20.91B total / 3.6B active

Origin: `unsloth/gpt-oss-20b-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 3060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `gpt-oss-default-llama` | Q8_K_XL | 12.28 GiB | Default Llama | primary |  | 131072 | 45 | 116.26 | 14.47 | 0.00% | 9209 stop |  |
| `gpt-oss-rtx-hot` | Q8_K_XL | 12.28 GiB | Hot-Cache | primary |  | 131072 |  | 234.69 | 35.23 | 78.51% | 7738 stop |  |
| `gpt-oss-rtx-primary-quadro-experts` | Q8_K_XL | 12.28 GiB | Hot-Cache | primary | secondary | 131072 |  | 150.05 | 17.39 | 98.40% | 7240 stop | Higher hot ratio, but the asymmetric second GPU is slower here. |

### GLM-4.7-Flash, 29.94B total / active parameter count not specified

Origin: `unsloth/GLM-4.7-Flash-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 3060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `glm4-flash-default-llama` | Q8_K_XL | 33.17 GiB | Default Llama | primary |  | 131072 | 46 | 50.37 | 10.31 | 0.00% | 8283 stop |  |
| `glm4-flash-rtx-hot` | Q8_K_XL | 33.17 GiB | Hot-Cache | primary |  | 131072 |  | 50.88 | 10.18 | 10.99% | 8679 stop | Very low hot ratio; hot-cache does not improve this profile. |
| `glm4-flash-rtx-primary-quadro-experts` | Q8_K_XL | 33.17 GiB | Hot-Cache | primary | secondary | 131072 |  | 49.07 | 8.23 | 31.50% | 6140 stop | Higher hot ratio, but the asymmetric second GPU is slower here. |

### GLM-4.7-Flash-REAP-23B-A3B, 23.00B total / 3B active

Origin: `unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 3060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `glm4moe-default-llama` | Q8_K_XL | 25.63 GiB | Default Llama | primary |  | 131072 | 45 | 66.16 | 10.47 | 0.00% | 8251 stop |  |
| `glm4moe-rtx-hot` | Q8_K_XL | 25.63 GiB | Hot-Cache | primary |  | 131072 |  | 65.50 | 10.40 | 12.55% | 7950 stop | Very low hot ratio; hot-cache does not improve this profile. |
| `glm4moe-rtx-primary-quadro-experts` | Q8_K_XL | 25.63 GiB | Hot-Cache | primary | secondary | 131072 |  | 62.47 | 8.43 | 34.28% | 5527 stop | Higher hot ratio, but the asymmetric second GPU is slower here. |

### Gemma-4-26B-A4B-it, 25.23B total / 4B active

Origin: `unsloth/gemma-4-26B-A4B-it-GGUF`

| Profile | Quantization | Disk size | Mode | RTX 3060 12GB eGPU | Quadro 4GB iGPU | Context | n-cpu-moe | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `gemma4-default-llama` | Q6_K_XL | 21.68 GiB | Default Llama | primary |  | 131072 | 22 | 92.78 | 18.38 | 0.00% | 7929 stop |  |
| `gemma4-rtx-hot-cache` | Q6_K_XL | 21.68 GiB | Hot-Cache | primary |  | 131072 |  | 104.64 | 27.39 | 71.49% | 8968 stop | Good RTX-only hot-cache win. |
| `gemma4-rtx-primary-quadro-experts` | Q6_K_XL | 21.68 GiB | Hot-Cache | primary | secondary | 131072 |  | 101.21 | 15.43 | 86.65% | 6338 stop | Higher hot ratio, but the asymmetric second GPU is slower here. |

---

# Upstream llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[Read the upstream llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
