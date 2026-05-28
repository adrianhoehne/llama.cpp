# hot-cache-force-cold-escape
| model                          |       size |     params | backend    | ngl |       cmoe | pf           | dev          | mhc          |  mhc_mib |  mhc_res | mhc_curve | mhc_w     |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | ------------ | ------------ | ------------ | -------: | -------: | --------: | --------- | --------------: | -------------------: |
| qwen35moe 35B.A3B Q6_K         |  29.65 GiB |    34.66 B | CUDA       |  99 |          1 | pp-bench-conversation-code.txt | CUDA0        | qwen36       |       -1 |     3000 |  0.700000 | flat      |           pp128 |         56.14 ± 1.98 |

build: 533e1177e (9402)
