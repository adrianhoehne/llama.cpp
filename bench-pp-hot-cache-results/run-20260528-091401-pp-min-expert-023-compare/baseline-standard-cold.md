ggml_cuda_init: found 2 CUDA devices (Total VRAM: 15868 MiB):
  Device 0: NVIDIA GeForce RTX 2060, compute capability 7.5, VMM: yes, VRAM: 11833 MiB
  Device 1: Quadro M1200, compute capability 5.0, VMM: yes, VRAM: 4035 MiB
llama-bench: benchmark 1/1: starting
llama-bench: benchmark 1/1: warmup prompt run
llama-bench: benchmark 1/1: prompt run 1/5
llama-bench: benchmark 1/1: prompt run 2/5
llama-bench: benchmark 1/1: prompt run 3/5
llama-bench: benchmark 1/1: prompt run 4/5
llama-bench: benchmark 1/1: prompt run 5/5
| model                          |       size |     params | backend    | ngl | pf           |  n_cpu_moe | dev          |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------ | ---------: | ------------ | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  29.65 GiB |    34.66 B | CUDA       |  99 | pp-bench-conversation-code.txt |         31 | CUDA0        |          pp3070 |         75.69 ± 0.01 |

build: 533e1177e (9402)
