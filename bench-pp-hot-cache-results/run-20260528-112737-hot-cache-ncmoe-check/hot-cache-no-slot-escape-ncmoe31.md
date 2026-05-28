| model                          |       size |     params | backend    | ngl | pf           |  n_cpu_moe | dev          | mhc          |  mhc_mib |  mhc_res | mhc_curve | mhc_w     |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------ | ---------: | ------------ | ------------ | -------: | -------: | --------: | --------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  29.65 GiB |    34.66 B | CUDA       |  99 | pp-bench-conversation-code.txt |         31 | CUDA0        | qwen36       |       -1 |     3000 |  0.700000 | flat      |          pp3070 |         47.86 ± 0.00 |

build: 533e1177e (9402)
