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

## Benchmark results

| Model | Quantization | Disk size | Mode | RTX 2060 12GB eGPU | Quadro 4GB iGPU | PP t/s | TG t/s | Hot ratio | Output | Comment |
|---|---|---:|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6-35B-A3B, 34.66B total / 3B active | Q8_K_XL | 38.5 GB | Default LLama | primary |  | 49.77 | 16.93 | 0.000 | 10000 |  |
| Qwen3.6-35B-A3B, 34.66B total / 3B active | Q8_K_XL | 38.5 GB | Hot-Cache | primary |  | 56.85 | 18.73 | 0.371 | 10000 | This is a low value hot ratio, only 37%, the higher this value is the faster PP and TG will be. |
| - | - | - | - | - | - | - | - | - | - | - |
| Mellum2-12B-A2.5B Thinking, 12.15B total / 2.5B active | Q8_0 | 12.9 GB | Default LLama | primary |  | 283.87 | 34.21 | 0.000 | 5556 |  |
| Mellum2-12B-A2.5B Thinking, 12.15B total / 2.5B active | Q8_0 | 12.9 GB | Hot-Cache | primary |  | 266.65 | 58.95 | 0.955 | 5399 |  |
| Mellum2-12B-A2.5B Thinking, 12.15B total / 2.5B active | Q8_0 | 12.9 GB | Hot-Cache | primary | secondary | 204.30 | 19.59 | 1.000 | 6078 |  |
| Mellum2-12B-A2.5B Thinking, 12.15B total / 2.5B active | Q8_0 | 12.9 GB | Hot-Cache | secondary | primary | 217.25 | 17.14 | 0.998 | 10000 |  |
| - | - | - | - | - | - | - | - | - | - | - |
| Mellum2-12B-A2.5B Base, 12.15B total / 2.5B active | BF16 | 24.3 GB | Hot-Cache | primary |  | 80.03 | 22.64 | 0.765 | 6744 |  |
| - | - | - | - | - | - | - | - | - | - | - |
| Mellum2-12B-A2.5B Thinking, 12.15B total / 2.5B active | Q4_K_M | 8.1 GB | Default LLama | primary |  | 1573.03 | 105.49 | 0.000 | 5418 | This is a good example when NOT to use this branch. If the model can live completely in VRAM then the overhead is too big. |
| Mellum2-12B-A2.5B Thinking, 12.15B total / 2.5B active | Q4_K_M | 8.1 GB | Hot-Cache | primary |  | 1090.49 | 99.27 | 1.000 | 3025 |  |
| - | - | - | - | - | - | - | - | - | - | - |
| Mellum2-12B-A2.5B Thinking, 12.15B total / 2.5B active | Q6_K | 10.9 GB | Hot-Cache | primary |  | 373.48 | 69.38 | 0.985 | 5974 |  |
| - | - | - | - | - | - | - | - | - | - | - |
| GPT-OSS-20B, 20.91B total / 3.6B active | Q8_K_XL | 13.2 GB | Default LLama | primary |  | 113.07 | 13.77 | 0.000 | 7497 |  |
| GPT-OSS-20B, 20.91B total / 3.6B active | Q8_K_XL | 13.2 GB | Hot-Cache | primary |  | 213.62 | 31.56 | 0.778 | 9940 |  |
| GPT-OSS-20B, 20.91B total / 3.6B active | Q8_K_XL | 13.2 GB | Hot-Cache | primary | secondary | 140.25 | 16.33 | 0.983 | 9052 |  |
| - | - | - | - | - | - | - | - | - | - | - |
| GLM-4.7-Flash-REAP-23B-A3B, 23.00B total / 3B active | Q8_K_XL | 27.5 GB | Default LLama | primary |  | 64.44 | 9.77 | 0.000 | 8360 |  |
| GLM-4.7-Flash-REAP-23B-A3B, 23.00B total / 3B active | Q8_K_XL | 27.5 GB | Hot-Cache | primary |  | 36.22 | 9.70 | 0.120 | 7283 |  |
| GLM-4.7-Flash-REAP-23B-A3B, 23.00B total / 3B active | Q8_K_XL | 27.5 GB | Hot-Cache | primary | secondary | 34.79 | 7.78 | 0.326 | 5703 |  |
| - | - | - | - | - | - | - | - | - | - | - |
| Gemma-4-26B-A4B-it, 25.23B total / 4B active | Q6_K_XL | 23.3 GB | Default LLama | primary |  | 90.35 | 16.92 | 0.000 | 8711 |  |
| Gemma-4-26B-A4B-it, 25.23B total / 4B active | Q6_K_XL | 23.3 GB | Hot-Cache | primary |  | 94.81 | 25.18 | 0.706 | 9538 |  |
| Gemma-4-26B-A4B-it, 25.23B total / 4B active | Q6_K_XL | 23.3 GB | Hot-Cache | primary | secondary | 88.14 | 14.37 | 0.858 | 5968 |  |
| Gemma-4-26B-A4B-it, 25.23B total / 4B active | Q6_K_XL | 23.3 GB | Hot-Cache | primary | secondary | 81.90 | 14.35 | 0.855 | 5922 |  |
| - | - | - | - | - | - | - | - | - | - | - |
| Qwen3-Coder-Next, 79.67B total / 3B active | IQ4_NL | 39.2 GB | Default LLama | primary |  | 69.71 | 9.51 | 0.000 | 7136 |  |
| Qwen3-Coder-Next, 79.67B total / 3B active | IQ4_NL | 39.2 GB | Hot-Cache | primary |  | 64.75 | 11.44 | 0.434 | 7865 |  |
| Qwen3-Coder-Next, 79.67B total / 3B active | IQ4_NL | 39.2 GB | Hot-Cache | primary | secondary | 41.67 | 5.36 | 0.543 | 7627 |  |

---

# Upstream llama.cpp

![llama](https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png)

[Read the upstream llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
