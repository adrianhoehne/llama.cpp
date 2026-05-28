ggml_cuda_init: found 2 CUDA devices (Total VRAM: 15868 MiB):
  Device 0: NVIDIA GeForce RTX 2060, compute capability 7.5, VMM: yes, VRAM: 11833 MiB
  Device 1: Quadro M1200, compute capability 5.0, VMM: yes, VRAM: 4035 MiB
llama-bench: benchmark 1/1: starting
llama-bench: benchmark 1/1: warmup prompt run
llama-bench: benchmark 1/1: prompt run 1/1
| model                          |       size |     params | backend    | ngl |       cmoe | pf           | dev          | mhc          |  mhc_mib | mhc_curve | mhc_w     |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------------ | ------------ | ------------ | -------: | --------: | --------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  29.65 GiB |    34.66 B | CUDA       |  99 |          1 | pp-bench-conversation-code.txt | CUDA0        | qwen36       |      384 |  0.700000 | flat      |          pp3070 |         58.20 ± 0.00 |

build: 533e1177e (9402)
